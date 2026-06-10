const HOP_BY_HOP_HEADERS = new Set([
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
  "host",
  "content-length"
]);

const RESPONSE_SKIP_HEADERS = new Set([
  "connection",
  "content-encoding",
  "content-length",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade"
]);

function getBackendBaseUrl() {
  const value = (process.env.BACKEND_BASE_URL || "").trim().replace(/\/+$/, "");
  if (!value) {
    throw new Error("BACKEND_BASE_URL is not configured");
  }
  if (!/^https?:\/\//i.test(value)) {
    throw new Error("BACKEND_BASE_URL must start with http:// or https://");
  }
  return value;
}

function readRequestBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", (chunk) => chunks.push(Buffer.from(chunk)));
    req.on("end", () => resolve(chunks.length ? Buffer.concat(chunks) : undefined));
    req.on("error", reject);
  });
}

function getForwardPath(req) {
  const incoming = new URL(req.url || "/", "https://vercel.local");
  let path = incoming.searchParams.get("path") || "";
  incoming.searchParams.delete("path");
  path = String(path).replace(/^\/+/, "");
  return "/" + path + (incoming.search ? incoming.search : "");
}

function buildForwardHeaders(req) {
  const headers = {};
  for (const [name, value] of Object.entries(req.headers || {})) {
    const lower = name.toLowerCase();
    if (HOP_BY_HOP_HEADERS.has(lower)) continue;
    if (Array.isArray(value)) {
      headers[name] = value.join(", ");
    } else if (typeof value !== "undefined") {
      headers[name] = String(value);
    }
  }
  headers["x-vercel-proxy"] = "xiaosita";
  return headers;
}

function writeCorsHeaders(res) {
  res.setHeader("access-control-allow-origin", "*");
  res.setHeader("access-control-allow-methods", "GET,POST,OPTIONS,HEAD");
  res.setHeader("access-control-allow-headers", "Content-Type,Range,Authorization");
  res.setHeader("access-control-expose-headers", "Content-Length,Content-Range,Accept-Ranges");
}

module.exports = async function handler(req, res) {
  writeCorsHeaders(res);
  if (req.method === "OPTIONS") {
    res.statusCode = 204;
    res.end();
    return;
  }

  let backendBase;
  try {
    backendBase = getBackendBaseUrl();
  } catch (error) {
    res.statusCode = 500;
    res.setHeader("content-type", "text/plain; charset=utf-8");
    res.end(error.message);
    return;
  }

  const forwardPath = getForwardPath(req);
  const upstreamUrl = backendBase + forwardPath;
  const method = req.method || "GET";
  const hasBody = !["GET", "HEAD"].includes(method.toUpperCase());

  try {
    const body = hasBody ? await readRequestBody(req) : undefined;
    const upstream = await fetch(upstreamUrl, {
      method,
      headers: buildForwardHeaders(req),
      body,
      redirect: "manual"
    });

    const buffer = Buffer.from(await upstream.arrayBuffer());
    res.statusCode = upstream.status;
    upstream.headers.forEach((value, name) => {
      const lower = name.toLowerCase();
      if (RESPONSE_SKIP_HEADERS.has(lower)) return;
      res.setHeader(name, value);
    });
    res.setHeader("content-length", String(buffer.length));
    res.end(buffer);
  } catch (error) {
    res.statusCode = 502;
    res.setHeader("content-type", "text/plain; charset=utf-8");
    res.end("Vercel proxy failed: " + (error && error.message ? error.message : String(error)));
  }
};
