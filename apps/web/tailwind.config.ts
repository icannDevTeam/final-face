import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
                extend: {
                    fontFamily: {
                        sans: ['"Plus Jakarta Sans"', 'sans-serif'],
                    },
                    colors: {
                        slate: {
                            850: '#151e2e',
                            900: '#0f172a',
                            950: '#020617',
                        },
                        brand: {
                            400: '#22d3ee', // Cyan
                            500: '#06b6d4',
                            600: '#0891b2',
                        }
                    },
                    animation: {
                        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                        'scan-line': 'scan 2s linear infinite',
                    },
                    keyframes: {
                        scan: {
                            '0%': { transform: 'translateY(-100%)' },
                            '100%': { transform: 'translateY(100%)' }
                        }
                    }
                }
            },
  plugins: [],
};
export default config;
