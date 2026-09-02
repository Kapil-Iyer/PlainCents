import { CATEGORIES, type Category } from "@/constants/categories";

/**
 * One fixed color per category (not by chart position), so a category reads
 * the same color on every chart and every render. HSL values chosen to stay
 * legible on both the light and dark surfaces defined in index.css — mid
 * lightness, moderate saturation, spread across the hue wheel.
 */
const PALETTE = [
  "217 91% 60%", // Food & Dining — primary blue
  "266 70% 62%", // Transport — violet
  "192 75% 45%", // Rent & Utilities — cyan
  "330 70% 60%", // Entertainment — pink
  "152 55% 42%", // Healthcare — green
  "38 92% 50%", // Shopping — amber
  "12 75% 55%", // Subscriptions — orange-red
  "218 10% 55%", // Other — neutral gray
] as const;

const CATEGORY_COLOR: Record<Category, string> = Object.fromEntries(
  CATEGORIES.map((category, i) => [category, `hsl(${PALETTE[i % PALETTE.length]})`]),
) as Record<Category, string>;

export function colorForCategory(category: string): string {
  return CATEGORY_COLOR[category as Category] ?? "hsl(218 10% 55%)";
}
