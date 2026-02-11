import joblib, random
pipe=joblib.load('models/v2.0.0/final_model.pkl')
features=pipe.get_booster().feature_names

min_p=10
best=None
for _ in range(20000):
    d={f:0.0 for f in features}
    d['ОХ перед ТП']=random.uniform(2.5,4.5)
    d['ЛПНП перед ТП']=random.uniform(1.0,3.0)
    d['ЛПВП перед ТП']=random.uniform(1.2,2.5)
    d['ТГ перед ТП']=random.uniform(0.5,1.5)
    d['Мочевая кислота перед ТП']=random.uniform(200,320)
    d['ЭХО ЛП перед ТП']=random.uniform(30,40)
    d['ЭХО КДР перед ТП']=random.uniform(40,55)
    d['ЭХО МЖП перед ТП']=random.uniform(8,11)
    d['ЭХО ЗС перед ТП']=random.uniform(8,11)
    d['ЭХО СДЛА перед ТП']=random.uniform(15,25)
    d['ЭХО ФВ перед ТП']=random.uniform(60,70)
    d['ЭХО ММЛЖ перед ТП']=random.uniform(90,140)
    d['ЭХО ИММЛЖ перед ТП']=random.uniform(80,120)
    d['ОТТ перед ТП']=0
    d['САД перед ТП']=random.uniform(90,120)
    d['ДАД перед ТП']=random.uniform(55,80)
    d['QRISK3']=random.uniform(0,3)
    d['healthy person risk']=random.uniform(0,2)
    d['relative risk']=random.uniform(0.1,0.8)
    d['qrisk age']=random.uniform(18,30)
    d['Пол_жен']=1; d['Пол_муж']=0
    # one-hot mutually exclusive
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
