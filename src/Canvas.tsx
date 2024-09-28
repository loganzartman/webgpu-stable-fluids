import {ForwardedRef, useEffect} from 'react';
import {forwardRef, useRef} from 'react';

import {useElementSize} from './useElementSize.ts';

export default forwardRef(function Canvas(
  {
    ariaLabel,
    onResize,
    style,
  }: {
    ariaLabel?: string;
    className?: string;
    onResize?: (size: [number, number]) => void;
    style?: React.CSSProperties;
  },
  ref: ForwardedRef<HTMLCanvasElement>,
) {
  const containerRef = useRef<HTMLDivElement>(null);
  const containerSize = useElementSize(containerRef);
  const onResizeRef = useRef(onResize);
  onResizeRef.current = onResize;

  useEffect(() => {
    onResizeRef.current?.(containerSize);
  }, [containerSize]);

  return (
    <div
      ref={containerRef}
      style={{
        position: 'relative',
        width: '100%',
        height: '100%',
        touchAction: 'none',
        userSelect: 'none',
        ...style
      }}
    >
      <canvas
        ref={ref}
        aria-label={ariaLabel}
        style={{
          position: 'absolute',
          width: '100%',
          height: '100%',
        }}
        width={containerSize[0]}
        height={containerSize[1]}
      />
    </div>
  );
});