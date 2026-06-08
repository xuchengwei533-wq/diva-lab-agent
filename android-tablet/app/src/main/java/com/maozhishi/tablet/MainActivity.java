package com.maozhishi.tablet;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.webkit.PermissionRequest;
import android.webkit.WebChromeClient;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;

import java.net.URLEncoder;

public class MainActivity extends Activity {
    private static final String PREFS_NAME = "MaoZhishiTablet";
    private static final String PREF_SERVER_INPUT = "ServerInput";
    private static final int REQUEST_WEB_PERMISSIONS = 1001;

    private WebView webView;
    private EditText serverInput;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWebPermissions();
        buildLayout();
        configureWebView();

        SharedPreferences prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        String lastInput = prefs.getString(PREF_SERVER_INPUT, "");
        if (lastInput == null || lastInput.trim().isEmpty()) {
            lastInput = BuildConfig.DEFAULT_PAGE_URL;
        }
        serverInput.setText(lastInput);
        loadFromInput(lastInput);
    }

    private void buildLayout() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);

        LinearLayout toolbar = new LinearLayout(this);
        toolbar.setOrientation(LinearLayout.HORIZONTAL);
        int padding = dp(8);
        toolbar.setPadding(padding, padding, padding, padding);

        serverInput = new EditText(this);
        serverInput.setSingleLine(true);
        serverInput.setHint("Backend host or full page URL");
        toolbar.addView(serverInput, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));

        Button loadButton = new Button(this);
        loadButton.setText("Load");
        loadButton.setOnClickListener(v -> loadFromInput(serverInput.getText().toString()));
        toolbar.addView(loadButton, new LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT));

        webView = new WebView(this);
        root.addView(toolbar, new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));
        root.addView(webView, new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f));
        setContentView(root);
    }

    private void configureWebView() {
        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setMediaPlaybackRequiresUserGesture(false);
        settings.setAllowFileAccess(true);
        settings.setAllowContentAccess(true);
        settings.setMixedContentMode(WebSettings.MIXED_CONTENT_ALWAYS_ALLOW);

        WebView.setWebContentsDebuggingEnabled(true);
        webView.setWebViewClient(new WebViewClient());
        webView.setWebChromeClient(new WebChromeClient() {
            @Override
            public void onPermissionRequest(PermissionRequest request) {
                runOnUiThread(() -> request.grant(request.getResources()));
            }
        });
    }

    private void loadFromInput(String rawInput) {
        String url = buildPageUrl(rawInput);
        serverInput.setText(rawInput);
        getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                .edit()
                .putString(PREF_SERVER_INPUT, rawInput)
                .apply();
        webView.loadUrl(url);
    }

    private String buildPageUrl(String rawInput) {
        String value = rawInput == null ? "" : rawInput.trim();
        if (value.isEmpty()) {
            value = BuildConfig.DEFAULT_PAGE_URL;
        }

        if (value.startsWith("http://") || value.startsWith("https://")) {
            return ensureDeploymentParams(value);
        }

        String hostPort = value
                .replace("http://", "")
                .replace("https://", "")
                .replace("/", "")
                .trim();
        String host = hostPort;
        String webPort = "8000";
        int colonIndex = hostPort.indexOf(":");
        if (colonIndex >= 0) {
            host = hostPort.substring(0, colonIndex);
            webPort = hostPort.substring(colonIndex + 1);
        }

        String encodedHost = encode(host);
        return "http://" + host + ":" + webPort + "/mao_demo.html"
                + "?apiHost=" + encodedHost
                + "&live2dHost=" + encodedHost
                + "&live2dPort=8010";
    }

    private String ensureDeploymentParams(String url) {
        Uri uri = Uri.parse(url);
        Uri.Builder builder = uri.buildUpon();
        String host = uri.getHost();
        if (host == null || host.isEmpty()) {
            return url;
        }
        if (uri.getQueryParameter("apiHost") == null) {
            builder.appendQueryParameter("apiHost", host);
        }
        if (uri.getQueryParameter("live2dHost") == null) {
            builder.appendQueryParameter("live2dHost", host);
        }
        if (uri.getQueryParameter("live2dPort") == null) {
            builder.appendQueryParameter("live2dPort", "8010");
        }
        return builder.build().toString();
    }

    private String encode(String value) {
        try {
            return URLEncoder.encode(value, "UTF-8");
        } catch (Exception ignored) {
            return value;
        }
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private void requestWebPermissions() {
        if (android.os.Build.VERSION.SDK_INT < 23) {
            return;
        }
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED
                || checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(
                    new String[] { Manifest.permission.RECORD_AUDIO, Manifest.permission.CAMERA },
                    REQUEST_WEB_PERMISSIONS
            );
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
}
