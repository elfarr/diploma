class FeatureField {
  const FeatureField({
    required this.name,
    required this.label,
    required this.min,
    required this.max,
    required this.unit,
    this.decimals = 1,
  });

  final String name;
  final String label;
  final double min;
  final double max;
  final String unit;
  final int decimals;
}

const demoFeatures = <FeatureField>[
  FeatureField(
    name: 'age',
    label: '�������',
    min: 18,
    max: 100,
    unit: '���',
    decimals: 0,
  ),
  FeatureField(
    name: 'sbp',
    label: '������������� ��������',
    min: 80,
    max: 220,
    unit: '�� ��. ��.',
  ),
  FeatureField(
    name: 'dbp',
    label: '�������������� ��������',
    min: 50,
    max: 140,
    unit: '�� ��. ��.',
  ),
  FeatureField(
    name: 'chol_total',
    label: '����� ����������',
    min: 2.0,
    max: 10.0,
    unit: '�����/�',
    decimals: 2,
  ),
  FeatureField(
    name: 'ldl',
    label: '����',
    min: 1.0,
    max: 8.0,
    unit: '�����/�',
    decimals: 2,
  ),
  FeatureField(
    name: 'bmi',
    label: '���',
    min: 15,
    max: 45,
    unit: '��/�?',
    decimals: 1,
  ),
  FeatureField(
    name: 'glucose',
    label: '�������',
    min: 3.0,
    max: 15.0,
    unit: '�����/�',
    decimals: 2,
  ),
];
