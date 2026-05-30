type Props = {
  className?: string;
  size?: number;
};

/**
 * Pulscale logo: a wave traversing a circle.
 *
 * The wave shape uses Q + chained T commands so each cycle reflects the
 * previous control point, producing a clean sinusoidal motion across the
 * circle's diameter without any external assets.
 */
export function Logo({ className = "", size = 32 }: Props) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-label="Pulscale logo"
    >
      <circle cx="16" cy="16" r="13.5" stroke="currentColor" strokeWidth="1.75" />
      <path
        d="M 5 16 Q 8 9 11 16 T 17 16 T 23 16 T 29 16"
        stroke="currentColor"
        strokeWidth="1.75"
        strokeLinecap="round"
        fill="none"
      />
    </svg>
  );
}
