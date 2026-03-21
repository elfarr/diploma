import 'report_download_stub.dart'
    if (dart.library.html) 'report_download_web.dart' as impl;

Future<void> downloadTextReport({
  required String filename,
  required String content,
}) {
  return impl.downloadTextReport(filename: filename, content: content);
}
