import 'dart:math' as math;

enum Qrisk3Sex { female, male }

enum Qrisk3Smoking {
  nonSmoker,
  exSmoker,
  lightSmoker,
  moderateSmoker,
  heavySmoker,
}

enum Qrisk3Diabetes { none, type1, type2 }

enum Qrisk3Ethnicity {
  whiteOrNotStated,
  indian,
  pakistani,
  bangladeshi,
  otherAsian,
  blackCaribbean,
  blackAfrican,
  chinese,
  otherEthnicGroup,
}

class Qrisk3Input {
  const Qrisk3Input({
    required this.age,
    required this.sex,
    required this.ethnicity,
    required this.smoking,
    required this.diabetes,
    required this.familyHistoryCvd,
    required this.chronicKidneyDisease,
    required this.atrialFibrillation,
    required this.bpTreatment,
    required this.migraine,
    required this.rheumatoidArthritis,
    required this.sle,
    required this.severeMentalIllness,
    required this.atypicalAntipsychotics,
    required this.steroids,
    required this.erectileDysfunction,
    required this.cholesterolHdlRatio,
    required this.systolicBp,
    required this.sbpStdDev,
    required this.heightCm,
    required this.weightKg,
    required this.townsend,
  });

  final double age;
  final Qrisk3Sex sex;
  final Qrisk3Ethnicity ethnicity;
  final Qrisk3Smoking smoking;
  final Qrisk3Diabetes diabetes;
  final bool familyHistoryCvd;
  final bool chronicKidneyDisease;
  final bool atrialFibrillation;
  final bool bpTreatment;
  final bool migraine;
  final bool rheumatoidArthritis;
  final bool sle;
  final bool severeMentalIllness;
  final bool atypicalAntipsychotics;
  final bool steroids;
  final bool erectileDysfunction;
  final double cholesterolHdlRatio;
  final double systolicBp;
  final double sbpStdDev;
  final double heightCm;
  final double weightKg;
  final double townsend;

  double get bmi {
    final hMeters = heightCm / 100.0;
    if (hMeters <= 0) {
      throw ArgumentError('Рост должен быть больше 0');
    }
    return weightKg / (hMeters * hMeters);
  }

  Qrisk3Input copyWith({
    double? age,
    Qrisk3Sex? sex,
    Qrisk3Ethnicity? ethnicity,
    Qrisk3Smoking? smoking,
    Qrisk3Diabetes? diabetes,
    bool? familyHistoryCvd,
    bool? chronicKidneyDisease,
    bool? atrialFibrillation,
    bool? bpTreatment,
    bool? migraine,
    bool? rheumatoidArthritis,
    bool? sle,
    bool? severeMentalIllness,
    bool? atypicalAntipsychotics,
    bool? steroids,
    bool? erectileDysfunction,
    double? cholesterolHdlRatio,
    double? systolicBp,
    double? sbpStdDev,
    double? heightCm,
    double? weightKg,
    double? townsend,
  }) {
    return Qrisk3Input(
      age: age ?? this.age,
      sex: sex ?? this.sex,
      ethnicity: ethnicity ?? this.ethnicity,
      smoking: smoking ?? this.smoking,
      diabetes: diabetes ?? this.diabetes,
      familyHistoryCvd: familyHistoryCvd ?? this.familyHistoryCvd,
      chronicKidneyDisease: chronicKidneyDisease ?? this.chronicKidneyDisease,
      atrialFibrillation: atrialFibrillation ?? this.atrialFibrillation,
      bpTreatment: bpTreatment ?? this.bpTreatment,
      migraine: migraine ?? this.migraine,
      rheumatoidArthritis: rheumatoidArthritis ?? this.rheumatoidArthritis,
      sle: sle ?? this.sle,
      severeMentalIllness: severeMentalIllness ?? this.severeMentalIllness,
      atypicalAntipsychotics:
          atypicalAntipsychotics ?? this.atypicalAntipsychotics,
      steroids: steroids ?? this.steroids,
      erectileDysfunction: erectileDysfunction ?? this.erectileDysfunction,
      cholesterolHdlRatio: cholesterolHdlRatio ?? this.cholesterolHdlRatio,
      systolicBp: systolicBp ?? this.systolicBp,
      sbpStdDev: sbpStdDev ?? this.sbpStdDev,
      heightCm: heightCm ?? this.heightCm,
      weightKg: weightKg ?? this.weightKg,
      townsend: townsend ?? this.townsend,
    );
  }
}

class Qrisk3Result {
  const Qrisk3Result({
    required this.qrisk3,
    required this.healthyPersonRisk,
    required this.relativeRisk,
    required this.qriskAge,
  });

  final double qrisk3;
  final double healthyPersonRisk;
  final double relativeRisk;
  final double qriskAge;
}

class Qrisk3Calculator {
  static const int _surv = 10;
  static const double minAge = 25.0;
  static const double maxAge = 84.0;
  static const double minSystolicBp = 70.0;
  static const double maxSystolicBp = 250.0;
  static const double minSbpStdDev = 0.0;
  static const double maxSbpStdDev = 40.0;
  static const double minHeightCm = 100.0;
  static const double maxHeightCm = 250.0;
  static const double minWeightKg = 20.0;
  static const double maxWeightKg = 300.0;
  static const double minTownsend = -15.0;
  static const double maxTownsend = 15.0;
  static const double minBmi = 15.0;
  static const double maxBmi = 60.0;
  static const double minCholesterolHdlRatio = 1.0;
  static const double maxCholesterolHdlRatio = 20.0;

  static Qrisk3Result calculate(Qrisk3Input input) {
    _validateInput(input);
    final qrisk3 = _calculateRisk(input);
    final healthy = _calculateRisk(_healthyPersonInput(input));
    final relative = healthy > 0 ? qrisk3 / healthy : double.infinity;
    final qage = _calculateQriskAge(input, qrisk3);
    return Qrisk3Result(
      qrisk3: qrisk3,
      healthyPersonRisk: healthy,
      relativeRisk: relative,
      qriskAge: qage,
    );
  }

  static void _validateInput(Qrisk3Input input) {
    if (!input.age.isFinite || input.age < minAge || input.age > maxAge) {
      throw ArgumentError('Возраст QRISK3 должен быть в диапазоне 25..84');
    }
    if (!input.cholesterolHdlRatio.isFinite ||
        input.cholesterolHdlRatio < minCholesterolHdlRatio ||
        input.cholesterolHdlRatio > maxCholesterolHdlRatio) {
      throw ArgumentError(
        'Соотношение холестерин/ЛПВП должно быть в диапазоне 1..20',
      );
    }
    if (!input.systolicBp.isFinite ||
        input.systolicBp < minSystolicBp ||
        input.systolicBp > maxSystolicBp) {
      throw ArgumentError('Систолическое АД должно быть в диапазоне 70..250');
    }
    if (!input.sbpStdDev.isFinite ||
        input.sbpStdDev < minSbpStdDev ||
        input.sbpStdDev > maxSbpStdDev) {
      throw ArgumentError(
        'Стандартное отклонение САД должно быть в диапазоне 0..40',
      );
    }
    if (!input.heightCm.isFinite ||
        input.heightCm < minHeightCm ||
        input.heightCm > maxHeightCm) {
      throw ArgumentError('Рост должен быть в диапазоне 100..250 см');
    }
    if (!input.weightKg.isFinite ||
        input.weightKg < minWeightKg ||
        input.weightKg > maxWeightKg) {
      throw ArgumentError('Вес должен быть в диапазоне 20..300 кг');
    }
    if (!input.townsend.isFinite ||
        input.townsend < minTownsend ||
        input.townsend > maxTownsend) {
      throw ArgumentError('Индекс Townsend должен быть в диапазоне -15..15');
    }
    final bmi = input.bmi;
    if (!bmi.isFinite || bmi < minBmi || bmi > maxBmi) {
      throw ArgumentError('BMI должен быть в диапазоне 15..60 кг/м²');
    }
  }

  static Qrisk3Input _healthyPersonInput(Qrisk3Input input) {
    return input.copyWith(
      smoking: Qrisk3Smoking.nonSmoker,
      diabetes: Qrisk3Diabetes.none,
      familyHistoryCvd: false,
      chronicKidneyDisease: false,
      atrialFibrillation: false,
      bpTreatment: false,
      migraine: false,
      rheumatoidArthritis: false,
      sle: false,
      severeMentalIllness: false,
      atypicalAntipsychotics: false,
      steroids: false,
      erectileDysfunction: false,
      cholesterolHdlRatio: 4.0,
      systolicBp: 125.0,
      sbpStdDev: 0.0,
      heightCm: input.sex == Qrisk3Sex.male ? 175.0 : 162.0,
      weightKg: input.sex == Qrisk3Sex.male ? 76.5625 : 65.61,
    );
  }

  static double _calculateQriskAge(Qrisk3Input input, double targetRisk) {
    double low = 25.0;
    double high = 95.0;
    final lowRisk = _calculateRisk(
      _healthyPersonInput(input).copyWith(age: low),
    );
    final highRisk = _calculateRisk(
      _healthyPersonInput(input).copyWith(age: high),
    );
    if (targetRisk <= lowRisk) {
      return low;
    }
    if (targetRisk >= highRisk) {
      return high;
    }

    for (var i = 0; i < 40; i++) {
      final mid = (low + high) / 2.0;
      final risk = _calculateRisk(
        _healthyPersonInput(input).copyWith(age: mid),
      );
      if (risk < targetRisk) {
        low = mid;
      } else {
        high = mid;
      }
    }
    return (low + high) / 2.0;
  }

  static double _calculateRisk(Qrisk3Input input) {
    return input.sex == Qrisk3Sex.female
        ? _femaleRisk(input)
        : _maleRisk(input);
  }

  static int _ethRiskIndex(Qrisk3Ethnicity ethnicity) {
    switch (ethnicity) {
      case Qrisk3Ethnicity.whiteOrNotStated:
        return 1;
      case Qrisk3Ethnicity.indian:
        return 2;
      case Qrisk3Ethnicity.pakistani:
        return 3;
      case Qrisk3Ethnicity.bangladeshi:
        return 4;
      case Qrisk3Ethnicity.otherAsian:
        return 5;
      case Qrisk3Ethnicity.blackCaribbean:
        return 6;
      case Qrisk3Ethnicity.blackAfrican:
        return 7;
      case Qrisk3Ethnicity.chinese:
        return 8;
      case Qrisk3Ethnicity.otherEthnicGroup:
        return 9;
    }
  }

  static int _smokeIndex(Qrisk3Smoking smoking) {
    switch (smoking) {
      case Qrisk3Smoking.nonSmoker:
        return 0;
      case Qrisk3Smoking.exSmoker:
        return 1;
      case Qrisk3Smoking.lightSmoker:
        return 2;
      case Qrisk3Smoking.moderateSmoker:
        return 3;
      case Qrisk3Smoking.heavySmoker:
        return 4;
    }
  }

  static double _femaleRisk(Qrisk3Input input) {
    const survivor = 0.988876402378082;
    const iEthRisk = [
      0.0,
      0.0,
      0.28040314332995425,
      0.562989941420754,
      0.29590000851116516,
      0.07278537987798255,
      -0.17072135508857317,
      -0.3937104331487497,
      -0.3263249528353027,
      -0.17127056883241784,
    ];
    const iSmoke = [
      0.0,
      0.13386833786546262,
      0.5620085801243854,
      0.6674959337750255,
      0.8494817764483085,
    ];

    final age = input.age;
    final bmi = input.bmi;
    final smokeCat = _smokeIndex(input.smoking);
    final ethrisk = _ethRiskIndex(input.ethnicity);
    final bType1 = input.diabetes == Qrisk3Diabetes.type1 ? 1 : 0;
    final bType2 = input.diabetes == Qrisk3Diabetes.type2 ? 1 : 0;

    final dage = age / 10.0;
    var age1 = math.pow(dage, -2).toDouble();
    var age2 = dage;
    final dbmi = bmi / 10.0;
    var bmi1 = math.pow(dbmi, -2).toDouble();
    var bmi2 = math.pow(dbmi, -2).toDouble() * math.log(dbmi);

    age1 -= 0.053274843841791;
    age2 -= 4.332503318786621;
    bmi1 -= 0.154946178197861;
    bmi2 -= 0.144462317228317;
    final rati = input.cholesterolHdlRatio - 3.47632646560669;
    final sbp = input.systolicBp - 123.13001251220703;
    final sbps5 = input.sbpStdDev - 9.002537727355957;
    final town = input.townsend - 0.392308831214905;

    var a = 0.0;
    a += iEthRisk[ethrisk];
    a += iSmoke[smokeCat];

    a += age1 * -8.138810924772619;
    a += age2 * 0.797333766896991;
    a += bmi1 * 0.2923609227546005;
    a += bmi2 * -4.1513300213837665;
    a += rati * 0.15338035820802554;
    a += sbp * 0.013131488407103424;
    a += sbps5 * 0.007889454101458609;
    a += town * 0.07722379058859011;

    a += (input.atrialFibrillation ? 1 : 0) * 1.5923354969269663;
    a += (input.atypicalAntipsychotics ? 1 : 0) * 0.25237642070115557;
    a += (input.steroids ? 1 : 0) * 0.5952072530460185;
    a += (input.migraine ? 1 : 0) * 0.301267260870345;
    a += (input.rheumatoidArthritis ? 1 : 0) * 0.21364803435181942;
    a += (input.chronicKidneyDisease ? 1 : 0) * 0.6519456949384583;
    a += (input.severeMentalIllness ? 1 : 0) * 0.12555308058820178;
    a += (input.sle ? 1 : 0) * 0.7588093865426769;
    a += (input.bpTreatment ? 1 : 0) * 0.50931593683423;
    a += bType1 * 1.7267977510537347;
    a += bType2 * 1.0688773244615468;
    a += (input.familyHistoryCvd ? 1 : 0) * 0.45445319020896213;

    a += age1 * (smokeCat == 1 ? 1 : 0) * -4.705716178585189;
    a += age1 * (smokeCat == 2 ? 1 : 0) * -2.7430383403573337;
    a += age1 * (smokeCat == 3 ? 1 : 0) * -0.8660808882939218;
    a += age1 * (smokeCat == 4 ? 1 : 0) * 0.9024156236971065;
    a += age1 * (input.atrialFibrillation ? 1 : 0) * 19.93803488954656;
    a += age1 * (input.steroids ? 1 : 0) * -0.9840804523593628;
    a += age1 * (input.migraine ? 1 : 0) * 1.7634979587873;
    a += age1 * (input.chronicKidneyDisease ? 1 : 0) * -3.5874047731694114;
    a += age1 * (input.sle ? 1 : 0) * 19.690303738638292;
    a += age1 * (input.bpTreatment ? 1 : 0) * 11.872809733921812;
    a += age1 * bType1 * -1.2444332714320747;
    a += age1 * bType2 * 6.86523420000096;
    a += age1 * bmi1 * 23.802623412141742;
    a += age1 * bmi2 * -71.184947692087;
    a += age1 * (input.familyHistoryCvd ? 1 : 0) * 0.9946780794043513;
    a += age1 * sbp * 0.034131842338615485;
    a += age1 * town * -1.030118080203564;
    a += age2 * (smokeCat == 1 ? 1 : 0) * -0.07558924464319303;
    a += age2 * (smokeCat == 2 ? 1 : 0) * -0.11951192874867074;
    a += age2 * (smokeCat == 3 ? 1 : 0) * -0.10366306397571923;
    a += age2 * (smokeCat == 4 ? 1 : 0) * -0.1399185359171839;
    a += age2 * (input.atrialFibrillation ? 1 : 0) * -0.0761826510111625;
    a += age2 * (input.steroids ? 1 : 0) * -0.12005364946742472;
    a += age2 * (input.migraine ? 1 : 0) * -0.06558691789869986;
    a += age2 * (input.chronicKidneyDisease ? 1 : 0) * -0.22688873086442507;
    a += age2 * (input.sle ? 1 : 0) * 0.07734794967901627;
    a += age2 * (input.bpTreatment ? 1 : 0) * 0.0009685782358817444;
    a += age2 * bType1 * -0.2872406462448895;
    a += age2 * bType2 * -0.09711225259069549;
    a += age2 * bmi1 * 0.5236995893366443;
    a += age2 * bmi2 * 0.04574419012232376;
    a += age2 * (input.familyHistoryCvd ? 1 : 0) * -0.07688505169842304;
    a += age2 * sbp * -0.0015082501423272358;
    a += age2 * town * -0.03159341467496233;

    return 100.0 * (1.0 - math.pow(survivor, math.exp(a)).toDouble());
  }

  static double _maleRisk(Qrisk3Input input) {
    const survivor = 0.977268040180206;
    const iEthRisk = [
      0.0,
      0.0,
      0.2771924876030828,
      0.4744636071493127,
      0.5296172991968937,
      0.03510015918629902,
      -0.3580789966932792,
      -0.4005648523216514,
      -0.41522792889830173,
      -0.26321348134749967,
    ];
    const iSmoke = [
      0.0,
      0.19128222863388983,
      0.5524158819264555,
      0.6383505302750607,
      0.7898381988185802,
    ];

    final age = input.age;
    final bmi = input.bmi;
    final smokeCat = _smokeIndex(input.smoking);
    final ethrisk = _ethRiskIndex(input.ethnicity);
    final bType1 = input.diabetes == Qrisk3Diabetes.type1 ? 1 : 0;
    final bType2 = input.diabetes == Qrisk3Diabetes.type2 ? 1 : 0;

    final dage = age / 10.0;
    var age1 = math.pow(dage, -1).toDouble();
    var age2 = math.pow(dage, 3).toDouble();
    final dbmi = bmi / 10.0;
    var bmi2 = math.pow(dbmi, -2).toDouble() * math.log(dbmi);
    var bmi1 = math.pow(dbmi, -2).toDouble();

    age1 -= 0.234766781330109;
    age2 -= 77.2840805053711;
    bmi1 -= 0.149176135659218;
    bmi2 -= 0.141913309693336;
    final rati = input.cholesterolHdlRatio - 4.300998687744141;
    final sbp = input.systolicBp - 128.5715789794922;
    final sbps5 = input.sbpStdDev - 8.756621360778809;
    final town = input.townsend - 0.52630490064621;

    var a = 0.0;
    a += iEthRisk[ethrisk];
    a += iSmoke[smokeCat];

    a += age1 * -17.839781666005575;
    a += age2 * 0.0022964880605765492;
    a += bmi1 * 2.4562776660536358;
    a += bmi2 * -8.301112231471135;
    a += rati * 0.1734019685632711;
    a += sbp * 0.012910126542553305;
    a += sbps5 * 0.010251914291290456;
    a += town * 0.033268201277287295;

    a += (input.atrialFibrillation ? 1 : 0) * 0.8820923692805466;
    a += (input.atypicalAntipsychotics ? 1 : 0) * 0.13046879855173513;
    a += (input.steroids ? 1 : 0) * 0.45485399750445543;
    a += (input.erectileDysfunction ? 1 : 0) * 0.22251859086705383;
    a += (input.migraine ? 1 : 0) * 0.25584178074159913;
    a += (input.rheumatoidArthritis ? 1 : 0) * 0.20970658013956567;
    a += (input.chronicKidneyDisease ? 1 : 0) * 0.7185326128827438;
    a += (input.severeMentalIllness ? 1 : 0) * 0.12133039882047164;
    a += (input.sle ? 1 : 0) * 0.4401572174457522;
    a += (input.bpTreatment ? 1 : 0) * 0.5165987108269547;
    a += bType1 * 1.2343425521675175;
    a += bType2 * 0.8594207143093222;
    a += (input.familyHistoryCvd ? 1 : 0) * 0.5405546900939016;

    a += age1 * (smokeCat == 1 ? 1 : 0) * -0.21011133933516346;
    a += age1 * (smokeCat == 2 ? 1 : 0) * 0.7526867644750319;
    a += age1 * (smokeCat == 3 ? 1 : 0) * 0.9931588755640579;
    a += age1 * (smokeCat == 4 ? 1 : 0) * 2.1331163414389076;
    a += age1 * (input.atrialFibrillation ? 1 : 0) * 3.4896675530623207;
    a += age1 * (input.steroids ? 1 : 0) * 1.1708133653489108;
    a += age1 * (input.erectileDysfunction ? 1 : 0) * -1.506400985745431;
    a += age1 * (input.migraine ? 1 : 0) * 2.349115987140244;
    a += age1 * (input.chronicKidneyDisease ? 1 : 0) * -0.5065671632722369;
    a += age1 * (input.bpTreatment ? 1 : 0) * 6.511458109853267;
    a += age1 * bType1 * 5.337986487800653;
    a += age1 * bType2 * 3.646181740622131;
    a += age1 * bmi1 * 31.004952956033886;
    a += age1 * bmi2 * -111.29157184391643;
    a += age1 * (input.familyHistoryCvd ? 1 : 0) * 2.7808628508531887;
    a += age1 * sbp * 0.018858524469865853;
    a += age1 * town * -0.1007554870063731;
    a += age2 * (smokeCat == 1 ? 1 : 0) * -0.0004985487027532612;
    a += age2 * (smokeCat == 2 ? 1 : 0) * -0.0007987563331738541;
    a += age2 * (smokeCat == 3 ? 1 : 0) * -0.000837061842662513;
    a += age2 * (smokeCat == 4 ? 1 : 0) * -0.0007840031915563729;
    a += age2 * (input.atrialFibrillation ? 1 : 0) * -0.0003499560834063605;
    a += age2 * (input.steroids ? 1 : 0) * -0.0002496045095297166;
    a += age2 * (input.erectileDysfunction ? 1 : 0) * -0.0011058218441227373;
    a += age2 * (input.migraine ? 1 : 0) * 0.00019896446041478631;
    a += age2 * (input.chronicKidneyDisease ? 1 : 0) * -0.0018325930166498813;
    a += age2 * (input.bpTreatment ? 1 : 0) * 0.0006383805310416501;
    a += age2 * bType1 * 0.0006409780808752897;
    a += age2 * bType2 * -0.00024695695588868315;
    a += age2 * bmi1 * 0.005038010235632203;
    a += age2 * bmi2 * -0.013074483002524319;
    a += age2 * (input.familyHistoryCvd ? 1 : 0) * -0.00024791809907396037;
    a += age2 * sbp * -0.00001271874191588457;
    a += age2 * town * -0.00009329964232327289;

    return 100.0 * (1.0 - math.pow(survivor, math.exp(a)).toDouble());
  }
}
