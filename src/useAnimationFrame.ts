import {useEffect, useRef} from 'react';

export const useAnimationFrame = (callback: () => void) => {
  const af = useRef<number | null>(null);

  useEffect(() => {
    const frame = () => {
      callback();
      af.current = requestAnimationFrame(frame);
    };

    af.current = requestAnimationFrame(frame);
    return () => {
      if (af.current !== null) cancelAnimationFrame(af.current);
    };
  }, [callback]);
};