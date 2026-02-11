from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import os
import datetime

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_FILE = os.path.join(BASE_DIR, "solar_model.pkl")

# Load Model
print("Loading model...")
try:
    model = joblib.load(MODEL_FILE)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # المدخلات الجديدة
        system_size = float(data.get('system_size', 2))  # حجم المنظومة بالكيلووات
        irradiation = float(data.get('irradiation', 0.8))
        temperature = float(data.get('temperature', 35))
        actual_power = data.get('actual_power')  # الإنتاج الفعلي اليومي (kWh)
        daily_consumption = data.get('daily_consumption', 0)  # الاستهلاك اليومي (kWh)
        
        print(f"DEBUG: system_size={system_size}, irradiation={irradiation}, temp={temperature}")
        print(f"DEBUG: actual_power={actual_power}, daily_consumption={daily_consumption}")
        
        try:
            daily_consumption = float(daily_consumption) if daily_consumption else 0
        except:
            daily_consumption = 0

        # ========================================
        # التحقق من صحة المدخلات (Validation)
        # ========================================
        validation_errors = []
        if system_size < 0:
            validation_errors.append("حجم النظام لا يمكن أن يكون سالباً.")
        if irradiation < 0:
            validation_errors.append("الإشعاع الشمسي لا يمكن أن يكون سالباً.")
        if irradiation > 1.2:
            validation_errors.append("الإشعاع الشمسي مرتفع جداً (أقصى حد 1.2 كيلو واط/م²).")
        
        # درجة الحرارة
        if temperature < -50: 
             validation_errors.append("درجة الحرارة منخفضة جداً وغير واقعية.")
        if temperature > 100:
             validation_errors.append("درجة الحرارة مرتفعة جداً (أقصى حد 100 درجة مئوية).")

        if daily_consumption < 0:
             validation_errors.append("الاستهلاك اليومي لا يمكن أن يكون سالباً.")
        
        if actual_power is not None:
             try:
                 actual_power = float(actual_power)
                 if actual_power < 0:
                     validation_errors.append("الإنتاج الفعلي لا يمكن أن يكون سالباً.")
             except ValueError:
                 validation_errors.append("الإنتاج الفعلي يجب أن يكون رقماً.")

        if validation_errors:
            return jsonify({'error': " | ".join(validation_errors)}), 400
        
        # الحصول على الساعة والشهر للموديل
        date_str = data.get('datetime')
        if date_str:
            dt = pd.to_datetime(date_str)
        else:
            dt = datetime.datetime.now()
        hour = dt.hour
        month = dt.month
            
        # ========================================
        # استخدام الموديل + المعادلة الفيزيائية
        # ========================================
        
        # معامل ساعات الذروة الشمسية في اليمن (Peak Sun Hours)
        PSH = 5.5
        
        # حجم المحطة التي تدرب عليها الموديل (تقريبي من بيانات Kaggle)
        KAGGLE_PLANT_SIZE = 4000  # kW
        
        # سعر الكهرباء في اليمن
        ELECTRICITY_PRICE_YER = 170
        
        # ========================================
        # الطريقة 1: استخدام الموديل (إذا متوفر)
        # ========================================
        model_efficiency = None
        
        if model is not None:
            try:
                # إعداد المدخلات للموديل
                features = pd.DataFrame([{
                    'IRRADIATION': irradiation,
                    'MODULE_TEMPERATURE': temperature,
                    'HOUR': hour,
                    'MONTH': month
                }])
                
                # الموديل يتوقع الطاقة اللحظية (kW) للمحطة الكبيرة
                predicted_power_kw = model.predict(features)[0]
                
                # حساب "نسبة الأداء" من الموديل
                # الإنتاج المثالي للمحطة الكبيرة = حجمها × إشعاع مثالي
                ideal_power = KAGGLE_PLANT_SIZE * irradiation
                
                if ideal_power > 0:
                    # نسبة الأداء = الفعلي ÷ المثالي
                    model_efficiency = min(1.0, max(0.3, predicted_power_kw / ideal_power))
                else:
                    model_efficiency = 0.5
                    
                print(f"DEBUG: Model predicted {predicted_power_kw:.2f} kW, efficiency = {model_efficiency:.2%}")
                
            except Exception as e:
                print(f"DEBUG: Model prediction failed: {e}")
                model_efficiency = None
        
        # ========================================
        # الطريقة 2: المعادلة الفيزيائية (Fallback)
        # ========================================
        
        # ثوابت فيزيائية آمنة (Safe Physical Constants)
        TEMPERATURE_COEFFICIENT = -0.0045  # معامل الحرارة (-0.45% لكل درجة مئوية فوق 25)
        STANDARD_TEMP = 25.0               # درجة الحرارة القياسية (STC)
        
        # معادلة القدرة الفيزيائية القياسية:
        # Power = Rated_Power * (Irradiance / 1000) * [1 + Coeff * (Temp - 25)]
        # نفترض أن الإشعاع المدخل (irradiation) بوحدة kW/m² (أي ما يعادل Irradiance/1000)
        
        # 1. حساب تأثير الحرارة
        temp_diff = temperature - STANDARD_TEMP
        temp_loss_factor = 1 + (TEMPERATURE_COEFFICIENT * temp_diff)
        
        # الكفاءة لا تزيد عن 100% ولا تقل عن 0% (نظرياً)
        # في الواقع، قد تزيد الكفاءة قليلاً إذا كانت الحرارة أقل من 25، لكن سنثبتها عند حد أقصى للحماية
        physics_performance_ratio = min(1.1, max(0.0, temp_loss_factor))
        
        # 2. حساب الإنتاج المتوقع فيزيائياً (كيلووات لحظي)
        # المعادلة: الحجم * الإشعاع * عامل الحرارة
        physics_power_kw = system_size * irradiation * physics_performance_ratio
        
        # تحويل القدرة اللحظية (kW) إلى كفاءة نسبية للمقارنة مع الموديل
        # الكفاءة هنا تعني نسبة الإنتاج الفعلي للإنتاج المعياري
        if (system_size * irradiation) > 0:
            physics_efficiency_ratio = physics_power_kw / (system_size * irradiation)
        else:
            physics_efficiency_ratio = 0.0

        # ========================================
        # دمج النتيجتين (متوسط مرجح)
        # ========================================
        
        if model_efficiency is not None:
            # استخدام الموديل كأساس (لأنه يرى أشياء لا تراها الفيزياء مثل الغبار الموسمي)
            # والفيزياء كعامل مساعد لضبط القيم الشاذة
            final_efficiency = (0.7 * model_efficiency) + (0.3 * physics_efficiency_ratio)
            prediction_source = "Hybrid: ML (70%) + Physics (30%)"
        else:
            final_efficiency = physics_efficiency_ratio
            prediction_source = "Physics Only (Standard Formula)"
        
        # الإنتاج المتوقع اليومي (kWh)
        # PSH (ساعات ذروة) * كفاءة النظام * حجم النظام
        # ملاحظة: المعادلة الفيزيائية أعلاه حسبت القدرة اللحظية.
        # لتحويلها ليومي، نستخدم PSH بدلاً من الإشعاع اللحظي في معادلة اليوم الكامل
        
        # المعادلة اليومية المحسنة:
        expected_daily = system_size * PSH * final_efficiency
        
        # ========================================
        # حساب الخسائر والتوصيات
        # ========================================
        
        status = "normal"
        power_loss = 0
        loss_percent = 0
        money_loss_daily = 0
        recommendation = "النظام يعمل بكفاءة ممتازة."
        
        # تحليل الإنتاج الفعلي (إذا تم إدخاله)
        if actual_power is not None and actual_power != "":
            actual_power = float(actual_power)
            
            # خسارة الغبار = المتوقع - الفعلي
            power_loss = max(0, expected_daily - actual_power)
            loss_percent = (power_loss / expected_daily) * 100 if expected_daily > 0 else 0
            
            # الخسارة المالية بسبب الغبار
            money_loss_daily = power_loss * ELECTRICITY_PRICE_YER
            
            # تصنيف حالة النظام
            if loss_percent > 5:
                if temperature > 50:
                    recommendation = "تحذير: الحرارة مرتفعة جداً! قد يسبب هذا نقاطاً ساخنة (Hotspots) تقلل الكفاءة."
                    status = "alert"
                elif loss_percent > 50:
                    recommendation = "خطر شديد! انخفاض هائل في الإنتاج. افحص: 1- الظلال 2- التوصيلات 3- الإنفرتر."
                    status = "alert"
                elif loss_percent > 30:
                    recommendation = "مشكلة واضحة! قد يكون السبب تظليل جزئي أو أوساخ كثيفة جداً. افحص الألواح."
                    status = "alert"
                elif loss_percent > 15:
                    recommendation = "تنبيه: الألواح تحتاج تنظيف. الغبار بدأ يؤثر بشكل ملحوظ على الإنتاج."
                    status = "warning"
                elif loss_percent > 5:
                    recommendation = "ملاحظة: بداية تراكم للغبار. يفضل جدولة تنظيف قريباً."
                    status = "warning"
                else:
                    recommendation = "الأداء ممتاز! الفاقد ضمن الحدود الطبيعية جداً."
                    status = "ok"
            else:
                recommendation = "ممتاز! النظام يعمل بكفاءة عالية (لا توجد خسارة تذكر)."
                status = "ok"
                
            daily_production_actual = actual_power
        else:
            # إذا لم يتم إدخال الإنتاج الفعلي، نستخدم المتوقع
            actual_power = None
            daily_production_actual = expected_daily
        
        # ========================================
        # تحليل الاستهلاك والعجز
        # ========================================
        
        deficit_kwh = 0
        cost_to_cover_deficit = 0
        
        if daily_consumption > 0:
            deficit_kwh = max(0, daily_consumption - daily_production_actual)
            cost_to_cover_deficit = deficit_kwh * ELECTRICITY_PRICE_YER
        
        # ========================================
        # بناء الاستجابة
        # ========================================
        
        response = {
            'predicted_power': round(expected_daily, 2),  # الإنتاج المتوقع اليومي kWh
            'actual_power': round(actual_power, 2) if actual_power is not None else None,
            'recommendation': recommendation,
            'status': status,
            'power_loss': round(power_loss, 2),  # خسارة الغبار kWh
            'loss_percent': round(loss_percent, 1),
            'consumption_analysis': {
                'daily_needed': daily_consumption,
                'daily_produced': round(daily_production_actual, 2),
                'deficit': round(deficit_kwh, 2),
                'cost_to_cover': int(round(cost_to_cover_deficit, 0))
            },
            'money_loss': {
                'hourly': 0,  # غير مستخدم الآن
                'daily': int(round(money_loss_daily, 0)),
                'monthly': int(round(money_loss_daily * 30, 0)),
                'currency': 'YER'
            },
            'details': {
                'system_size': system_size,
                'efficiency': round(final_efficiency * 100, 1),
                'psh': PSH,
                'prediction_source': prediction_source,
                'input_irradiation': irradiation,
                'input_temp': temperature
            }
        }
        
        print(f"DEBUG: Response = {response}")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
