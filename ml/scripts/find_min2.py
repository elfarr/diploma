import joblib, random
pipe=joblib.load('models/v2.0.0/final_model.pkl')
features=pipe.get_booster().feature_names

min_p=10
best=None
for _ in range(12000):
    d={f:0.0 for f in features}
    d['ОХ перед ТП']=random.uniform(2.0,4.0)
    d['ЛПНП перед ТП']=random.uniform(0.5,2.0)
    d['ЛПВП перед ТП']=random.uniform(2.0,3.0)
    d['ТГ перед ТП']=random.uniform(0.3,1.0)
    d['Мочевая кислота перед ТП']=random.uniform(180,280)
    d['ЭХО ЛП перед ТП']=random.uniform(25,35)
    d['ЭХО КДР перед ТП']=random.uniform(38,50)
    d['ЭХО МЖП перед ТП']=random.uniform(7,10)
    d['ЭХО ЗС перед ТП']=random.uniform(7,10)
    d['ЭХО СДЛА перед ТП']=random.uniform(12,20)
    d['ЭХО ФВ перед ТП']=random.uniform(65,75)
    d['ЭХО ММЛЖ перед ТП']=random.uniform(80,120)
    d['ЭХО ИММЛЖ перед ТП']=random.uniform(70,100)
    d['ОТТ перед ТП']=0
    d['САД перед ТП']=random.uniform(90,110)
    d['ДАД перед ТП']=random.uniform(50,70)
    d['QRISK3']=random.uniform(0,1)
    d['healthy person risk']=random.uniform(0,1)
    d['relative risk']=random.uniform(0.05,0.3)
    d['qrisk age']=random.uniform(18,25)
    d['Пол_жен']=1; d['Пол_муж']=0
    for name in features:
        if '_нет' in name:
            d[name]=1
        if '_есть' in name:
            d[name]=0
    x=[d.get(f,0.0) for f in features]
    p=float(pipe.predict_proba([x])[0,1])
    if p<min_p:
        min_p=p; best=d.copy()
print('min',min_p)
print({k:v for k,v in best.items() if k not in ('Пол_жен','Пол_муж') and not k.endswith('_нет') and not k.endswith('_есть')})
