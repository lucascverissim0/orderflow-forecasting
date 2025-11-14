/** @type {import('next').NextConfig} */

const API_URL = "https://redesigned-funicular-r4v4jrrxxq67fpr47-8000.app.github.dev";

const nextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${API_URL}/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
