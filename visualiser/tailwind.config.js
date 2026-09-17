/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./app/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0f1117",
        panel: "#171a23",
        panel2: "#1e2230",
        edge: "#2a2f3d",
        muted: "#9aa2b4",
        accent: "#5b8def",
        accent2: "#7c5cff",
      },
      fontFamily: {
        sans: ["-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "Helvetica", "Arial", "sans-serif"],
      },
    },
  },
  plugins: [],
};
