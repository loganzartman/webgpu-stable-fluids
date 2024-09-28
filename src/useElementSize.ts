import {useLayoutEffect, useState} from 'react';

export const useElementSize = (
  ref: React.RefObject<Element>,
): [number, number] => {
  const [elementSize, setElementSize] = useState<[number, number]>([256, 256]);

  useLayoutEffect(() => {
    const el = ref.current;
    if (el) {
      const observer = new ResizeObserver((entries) => {
        const rect = entries[0].contentRect;
        setElementSize([
          rect.width * window.devicePixelRatio,
          rect.height * window.devicePixelRatio,
        ]);
      });
      observer.observe(el);
      return () => observer.unobserve(el);
    }
  }, [ref]);

  return elementSize;
};