import 'dart:math' as math;

const double glucoseFactor = 18.0;
const double cholesterolFactor = 38.67; 
const double triglycerideFactor = 88.57; 
const double uricAcidFactor = 59.48; 

double mgDlToMmolL(double v, {required double factor}) {
  return v / factor;
}

double mmolLToMgDl(double v, {required double factor}) {
  return v * factor;
}

double kpaToMmhg(double v) {
  return v * 7.50062;
}

double mmhgToKpa(double v) {
  return v / 7.50062;
}

double umolLToMgDl(double v, {required double factor}) {
  return v / factor;
}

double mgDlToUmolL(double v, {required double factor}) {
  return v * factor;
}

String formatNum(double v, {int digits = 2}) {
  final factor = math.pow(10, digits).toDouble();
  final floored = (v * factor).floorToDouble() / factor;
  final s = floored.toStringAsFixed(digits);
  return s.replaceFirst(RegExp(r'\.?0+$'), '');
}
