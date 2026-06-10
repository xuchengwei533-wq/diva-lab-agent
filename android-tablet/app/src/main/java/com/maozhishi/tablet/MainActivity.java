package com.maozhishi.tablet;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioAttributes;
import android.media.AudioManager;
import android.media.MediaPlayer;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Base64;
import android.view.View;
import android.webkit.JavascriptInterface;
import android.webkit.PermissionRequest;
import android.webkit.WebChromeClient;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;

import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class MainActivity extends Activity {
    private static final String PREFS_NAME = "MaoZhishiTablet";
    private static final String PREF_SERVER_INPUT = "ServerInput";
    private static final String PREF_DEFAULT_PAGE_URL = "DefaultPageUrl";
    private static final int REQUEST_WEB_PERMISSIONS = 1001;

    private WebView webView;
    private EditText serverInput;
    private NativeVoiceBridge nativeVoiceBridge;
    private NativeAudioBridge nativeAudioBridge;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        SharedPreferences prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        String lastInput = prefs.getString(PREF_SERVER_INPUT, "");
        String savedDefaultUrl = prefs.getString(PREF_DEFAULT_PAGE_URL, "");
        if (shouldUseCurrentDefault(lastInput, savedDefaultUrl)) {
            lastInput = BuildConfig.DEFAULT_PAGE_URL;
        }
        requestWebPermissions(buildPageUrl(lastInput));
        applyFullscreenChrome();
        buildLayout();
        configureWebView();

        loadFromInput(lastInput);
    }

    private void buildLayout() {
        webView = new WebView(this);
        setContentView(webView, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.MATCH_PARENT
        ));
    }

    private void configureWebView() {
        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setDatabaseEnabled(true);
        settings.setLoadsImagesAutomatically(true);
        settings.setCacheMode(WebSettings.LOAD_DEFAULT);
        settings.setMediaPlaybackRequiresUserGesture(false);
        settings.setAllowFileAccess(true);
        settings.setAllowContentAccess(true);
        settings.setMixedContentMode(WebSettings.MIXED_CONTENT_ALWAYS_ALLOW);

        WebView.setWebContentsDebuggingEnabled(BuildConfig.DEBUG);
        webView.setWebViewClient(new WebViewClient());
        webView.setWebChromeClient(new WebChromeClient() {
            @Override
            public void onPermissionRequest(PermissionRequest request) {
                runOnUiThread(() -> request.grant(request.getResources()));
            }
        });
        nativeVoiceBridge = new NativeVoiceBridge(this, webView);
        webView.addJavascriptInterface(nativeVoiceBridge, "AndroidVoice");
        nativeAudioBridge = new NativeAudioBridge(webView);
        webView.addJavascriptInterface(nativeAudioBridge, "AndroidAudio");
    }

    private void loadFromInput(String rawInput) {
        String url = buildPageUrl(rawInput);
        if (serverInput != null) {
            serverInput.setText(rawInput);
        }
        getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                .edit()
                .putString(PREF_SERVER_INPUT, rawInput)
                .putString(PREF_DEFAULT_PAGE_URL, BuildConfig.DEFAULT_PAGE_URL)
                .apply();
        webView.loadUrl(url);
    }

    private void applyFullscreenChrome() {
        getWindow().getDecorView().setSystemUiVisibility(
                View.SYSTEM_UI_FLAG_FULLSCREEN
                        | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                        | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                        | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                        | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                        | View.SYSTEM_UI_FLAG_LAYOUT_STABLE
        );
    }

    private boolean shouldUseCurrentDefault(String savedInput, String savedDefaultUrl) {
        if (savedInput == null || savedInput.trim().isEmpty()) {
            return true;
        }
        String value = savedInput.trim().toLowerCase(Locale.ROOT);
        if (value.contains("192.168.")
                || value.contains("10.")
                || value.contains("172.16.")
                || value.contains("172.17.")
                || value.contains("172.18.")
                || value.contains("172.19.")
                || value.contains("172.20.")
                || value.contains("172.21.")
                || value.contains("172.22.")
                || value.contains("172.23.")
                || value.contains("172.24.")
                || value.contains("172.25.")
                || value.contains("172.26.")
                || value.contains("172.27.")
                || value.contains("172.28.")
                || value.contains("172.29.")
                || value.contains("172.30.")
                || value.contains("172.31.")
                || value.contains("localhost")
                || value.contains("127.0.0.1")) {
            return true;
        }
        return savedDefaultUrl != null
                && !savedDefaultUrl.trim().isEmpty()
                && !BuildConfig.DEFAULT_PAGE_URL.equals(savedDefaultUrl)
                && savedInput.equals(savedDefaultUrl);
    }

    private String buildPageUrl(String rawInput) {
        String value = rawInput == null ? "" : rawInput.trim();
        if (value.isEmpty()) {
            value = BuildConfig.DEFAULT_PAGE_URL;
        }

        String lowerValue = value.toLowerCase(Locale.ROOT);
        if (lowerValue.startsWith("http://") || lowerValue.startsWith("https://")) {
            return ensureDeploymentParams(value);
        }

        Uri uri = Uri.parse("http://" + value);
        String host = uri.getHost();
        if (host == null || host.trim().isEmpty()) {
            return BuildConfig.DEFAULT_PAGE_URL;
        }

        String webPort = uri.getPort() > 0 ? String.valueOf(uri.getPort()) : "8000";
        String path = uri.getPath();
        if (path == null || path.trim().isEmpty() || "/".equals(path)) {
            path = "/tablet_legacy.html";
        }

        String hostForAuthority = host.contains(":") && !host.startsWith("[") ? "[" + host + "]" : host;
        Uri.Builder builder = new Uri.Builder()
                .scheme("http")
                .encodedAuthority(hostForAuthority + ":" + webPort)
                .encodedPath(path);
        if (uri.getEncodedQuery() != null) {
            builder.encodedQuery(uri.getEncodedQuery());
        }
        return ensureDeploymentParams(builder.build().toString());
    }

    private String ensureDeploymentParams(String url) {
        Uri uri = Uri.parse(url);
        Uri.Builder builder = uri.buildUpon();
        String host = uri.getHost();
        if (host == null || host.isEmpty()) {
            return url;
        }
        boolean singleOrigin = isTruthyQueryParam(uri, "singleOrigin")
                || isTruthyQueryParam(uri, "publicMode");
        if (uri.getQueryParameter("singleOrigin") == null) {
            builder.appendQueryParameter("singleOrigin", "1");
            singleOrigin = true;
        }
        if (!singleOrigin && uri.getQueryParameter("apiHost") == null) {
            builder.appendQueryParameter("apiHost", host);
        }
        if (!singleOrigin && uri.getQueryParameter("live2dHost") == null) {
            builder.appendQueryParameter("live2dHost", host);
        }
        if (!singleOrigin && uri.getQueryParameter("live2dPort") == null) {
            builder.appendQueryParameter("live2dPort", "8010");
        }
        if (uri.getQueryParameter("legacy") == null) {
            builder.appendQueryParameter("legacy", "1");
        }
        if (uri.getQueryParameter("lite") == null) {
            builder.appendQueryParameter("lite", "1");
        }
        if (uri.getQueryParameter("renderMode") == null) {
            builder.appendQueryParameter("renderMode", "gif");
        }
        if (uri.getQueryParameter("disableFace") == null) {
            builder.appendQueryParameter("disableFace", "1");
        }
        return builder.build().toString();
    }

    private boolean isTruthyQueryParam(Uri uri, String name) {
        String value = uri.getQueryParameter(name);
        if (value == null) {
            return false;
        }
        String normalized = value.trim().toLowerCase(Locale.ROOT);
        return "1".equals(normalized)
                || "true".equals(normalized)
                || "yes".equals(normalized)
                || "on".equals(normalized);
    }

    private boolean shouldRequestCamera(String pageUrl) {
        Uri uri = Uri.parse(pageUrl);
        if (isTruthyQueryParam(uri, "disableFace")
                || isTruthyQueryParam(uri, "legacy")
                || isTruthyQueryParam(uri, "lite")) {
            return false;
        }
        String renderMode = uri.getQueryParameter("renderMode");
        return renderMode == null || !"gif".equalsIgnoreCase(renderMode.trim());
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private void requestWebPermissions(String pageUrl) {
        if (android.os.Build.VERSION.SDK_INT < 23) {
            return;
        }
        List<String> missingPermissions = new ArrayList<>();
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            missingPermissions.add(Manifest.permission.RECORD_AUDIO);
        }
        if (shouldRequestCamera(pageUrl)
                && checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            missingPermissions.add(Manifest.permission.CAMERA);
        }
        if (!missingPermissions.isEmpty()) {
            requestPermissions(missingPermissions.toArray(new String[0]), REQUEST_WEB_PERMISSIONS);
        }
    }

    @Override
    public void onBackPressed() {
        if (webView != null && webView.canGoBack()) {
            webView.goBack();
            return;
        }
        super.onBackPressed();
    }

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        if (hasFocus) {
            applyFullscreenChrome();
        }
    }

    @Override
    protected void onDestroy() {
        if (nativeVoiceBridge != null) {
            nativeVoiceBridge.stop();
        }
        if (nativeAudioBridge != null) {
            nativeAudioBridge.stop();
        }
        super.onDestroy();
    }

    public static class NativeAudioBridge {
        private final WebView webView;
        private final Handler mainHandler = new Handler(Looper.getMainLooper());
        private MediaPlayer mediaPlayer;

        NativeAudioBridge(WebView webView) {
            this.webView = webView;
        }

        @JavascriptInterface
        public boolean isAvailable() {
            return true;
        }

        @JavascriptInterface
        public void playUrl(String url) {
            final String audioUrl = url == null ? "" : url.trim();
            if (audioUrl.isEmpty()) {
                postError("音频地址为空");
                return;
            }
            mainHandler.post(() -> playOnMainThread(audioUrl));
        }

        @JavascriptInterface
        public void stop() {
            mainHandler.post(this::stopPlayer);
        }

        private void playOnMainThread(String url) {
            try {
                stopPlayer();
                MediaPlayer player = new MediaPlayer();
                mediaPlayer = player;
                if (Build.VERSION.SDK_INT >= 21) {
                    player.setAudioAttributes(new AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_MEDIA)
                            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                            .build());
                } else {
                    player.setAudioStreamType(AudioManager.STREAM_MUSIC);
                }
                player.setOnPreparedListener(mp -> {
                    postStatus("正在播放语音...");
                    mp.start();
                });
                player.setOnCompletionListener(mp -> {
                    stopPlayer();
                    postEnded();
                });
                player.setOnErrorListener((mp, what, extra) -> {
                    stopPlayer();
                    postError("MediaPlayer error " + what + "/" + extra);
                    return true;
                });
                player.setDataSource(url);
                postStatus("正在加载语音...");
                player.prepareAsync();
            } catch (Exception error) {
                stopPlayer();
                postError(error.getMessage() == null ? "原生播放失败" : error.getMessage());
            }
        }

        private void stopPlayer() {
            MediaPlayer player = mediaPlayer;
            mediaPlayer = null;
            if (player == null) {
                return;
            }
            try {
                if (player.isPlaying()) {
                    player.stop();
                }
            } catch (Exception ignored) {
            }
            try {
                player.release();
            } catch (Exception ignored) {
            }
        }

        private void postStatus(String status) {
            runJs("window.onNativeAudioStatus&&window.onNativeAudioStatus('" + escapeJs(status) + "')");
        }

        private void postEnded() {
            runJs("window.onNativeAudioEnded&&window.onNativeAudioEnded()");
        }

        private void postError(String error) {
            runJs("window.onNativeAudioError&&window.onNativeAudioError('" + escapeJs(error) + "')");
        }

        private void runJs(String script) {
            mainHandler.post(() -> {
                if (Build.VERSION.SDK_INT >= 19) {
                    webView.evaluateJavascript(script, null);
                } else {
                    webView.loadUrl("javascript:" + script);
                }
            });
        }

        private String escapeJs(String value) {
            if (value == null) {
                return "";
            }
            return value
                    .replace("\\", "\\\\")
                    .replace("'", "\\'")
                    .replace("\r", "\\r")
                    .replace("\n", "\\n");
        }
    }

    public static class NativeVoiceBridge {
        private static final int SAMPLE_RATE = 16000;
        private static final int CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO;
        private static final int AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT;
        private static final int MAX_RECORD_SECONDS = 30;

        private final Activity activity;
        private final WebView webView;
        private final Handler mainHandler = new Handler(Looper.getMainLooper());
        private final Object lock = new Object();

        private volatile boolean recording;
        private AudioRecord audioRecord;
        private Thread recordingThread;

        NativeVoiceBridge(Activity activity, WebView webView) {
            this.activity = activity;
            this.webView = webView;
        }

        @JavascriptInterface
        public boolean isAvailable() {
            return true;
        }

        @JavascriptInterface
        public void start() {
            synchronized (lock) {
                if (recording) {
                    postStatus("录音中，再点麦克风停止");
                    return;
                }
                if (Build.VERSION.SDK_INT >= 23
                        && activity.checkSelfPermission(Manifest.permission.RECORD_AUDIO)
                        != PackageManager.PERMISSION_GRANTED) {
                    activity.runOnUiThread(() -> activity.requestPermissions(
                            new String[]{Manifest.permission.RECORD_AUDIO},
                            REQUEST_WEB_PERMISSIONS
                    ));
                    postResult("", "未授予麦克风权限，请允许后重试");
                    return;
                }
                recording = true;
                recordingThread = new Thread(this::recordLoop, "MaoNativeVoiceRecorder");
                recordingThread.start();
            }
        }

        @JavascriptInterface
        public void stop() {
            recording = false;
            AudioRecord current = audioRecord;
            if (current != null) {
                try {
                    current.stop();
                } catch (Exception ignored) {
                }
            }
        }

        private void recordLoop() {
            ByteArrayOutputStream pcm = new ByteArrayOutputStream();
            int minBuffer = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT);
            int bufferSize = Math.max(minBuffer, SAMPLE_RATE);
            byte[] buffer = new byte[bufferSize];
            int maxBytes = SAMPLE_RATE * 2 * MAX_RECORD_SECONDS;

            try {
                audioRecord = buildAudioRecord(bufferSize, MediaRecorder.AudioSource.VOICE_RECOGNITION);
                if (audioRecord == null || audioRecord.getState() != AudioRecord.STATE_INITIALIZED) {
                    releaseRecorder();
                    audioRecord = buildAudioRecord(bufferSize, MediaRecorder.AudioSource.MIC);
                }
                if (audioRecord == null || audioRecord.getState() != AudioRecord.STATE_INITIALIZED) {
                    throw new IllegalStateException("AudioRecord 初始化失败");
                }

                audioRecord.startRecording();
                postStatus("原生录音中，再点麦克风停止");
                while (recording && pcm.size() < maxBytes) {
                    int read = audioRecord.read(buffer, 0, buffer.length);
                    if (read > 0) {
                        pcm.write(buffer, 0, read);
                    }
                }
                if (pcm.size() < SAMPLE_RATE / 2) {
                    postResult("", "录音太短，请重试");
                    return;
                }
                postStatus("识别中...");
                postResult(Base64.encodeToString(pcm.toByteArray(), Base64.NO_WRAP), "");
            } catch (Exception error) {
                postResult("", error.getMessage() == null ? "录音失败" : error.getMessage());
            } finally {
                recording = false;
                releaseRecorder();
            }
        }

        @SuppressWarnings("MissingPermission")
        private AudioRecord buildAudioRecord(int bufferSize, int source) {
            try {
                return new AudioRecord(source, SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT, bufferSize);
            } catch (Exception ignored) {
                return null;
            }
        }

        private void releaseRecorder() {
            AudioRecord current = audioRecord;
            audioRecord = null;
            if (current == null) {
                return;
            }
            try {
                current.release();
            } catch (Exception ignored) {
            }
        }

        private void postStatus(String status) {
            runJs("window.onNativeVoiceStatus&&window.onNativeVoiceStatus('" + escapeJs(status) + "')");
        }

        private void postResult(String base64Pcm, String error) {
            runJs("window.onNativeVoiceResult&&window.onNativeVoiceResult('"
                    + escapeJs(base64Pcm) + "','"
                    + escapeJs(error) + "')");
        }

        private void runJs(String script) {
            mainHandler.post(() -> {
                if (Build.VERSION.SDK_INT >= 19) {
                    webView.evaluateJavascript(script, null);
                } else {
                    webView.loadUrl("javascript:" + script);
                }
            });
        }

        private String escapeJs(String value) {
            if (value == null) {
                return "";
            }
            return value
                    .replace("\\", "\\\\")
                    .replace("'", "\\'")
                    .replace("\r", "\\r")
                    .replace("\n", "\\n");
        }
    }
}
