import 'dart:convert';
import 'dart:html' as html;
import 'dart:typed_data';

Future<void> downloadTextReport({
  required String filename,
  required String content,
}) async {
  final bytes = Uint8List.fromList(utf8.encode(content));
  final blob = html.Blob(<Object>[bytes], 'text/plain;charset=utf-8');
  final url = html.Url.createObjectUrlFromBlob(blob);
  final anchor = html.AnchorElement(href: url)
    ..download = filename
    ..style.display = 'none';

  html.document.body?.children.add(anchor);
  anchor.click();
  anchor.remove();
  html.Url.revokeObjectUrl(url);
}
