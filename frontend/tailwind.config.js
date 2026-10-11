/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: '#0B1215',
        surface: '#111B1F',
        raised: '#16242A',
        border: '#22353C',
        text: '#E6EEF0',
        muted: '#8CA3AB',
        emerald: {
          500: '#10B981',
          accent: '#10B981',
        },
        cyan: {
          400: '#38BDF8',
          accent: '#38BDF8',
        },
        amber: {
          500: '#F59E0B',
        },
        rose: {
          500: '#F43F5E',
        },
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      borderRadius: {
        hud: '12px',
      },
    },
  },
  plugins: [],
}
