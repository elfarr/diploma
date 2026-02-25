import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

import 'package:kidney_tx_app/main.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('predict flow with undetermined preset', (tester) async {
    await tester.pumpWidget(const MyApp());

    final undetBtn = find.widgetWithText(ElevatedButton, 'Undetermined');
    expect(undetBtn, findsOneWidget);
    await tester.tap(undetBtn);
    await tester.pumpAndSettle();


    final predictBtn = find.widgetWithText(ElevatedButton, 'Predict');
    expect(predictBtn, findsOneWidget);
    await tester.tap(predictBtn);


    var found = false;
    for (var i = 0; i < 60; i++) {
      await tester.pump(const Duration(seconds: 1));
      final hasClass = find.textContaining('Class:').evaluate().isNotEmpty;
      final hasProb = find.textContaining('p_cal').evaluate().isNotEmpty;
      final hasUndet =
          find.textContaining('undetermined', findRichText: true).evaluate().isNotEmpty;
      if (hasClass && hasProb) {
        found = true;
        expect(hasUndet, isTrue,
            reason: 'Ожидаемый класс == undetermined');
        break;
      }
    }
    expect(found, isTrue, reason: 'Timed out');
  });
}
