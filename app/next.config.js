/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,

  // In development, allow CORS requests to FastAPI (port 8000)
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/:path*"
      }
    ];
  }
};

module.exports = nextConfig;
