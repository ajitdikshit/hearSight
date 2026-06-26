import { useRef, useState } from "react";

export function useMicrophone() {
  const [active, setActive] = useState(false);
  const streamRef = useRef(null);
  const sourceRef = useRef(null);
  const ctxRef = useRef(null);

  const start = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      // CRITICAL: Force the browser to downsample to 16kHz
      const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      const source = ctx.createMediaStreamSource(stream);
      
      streamRef.current = stream;
      sourceRef.current = source;
      ctxRef.current = ctx;
      setActive(true);
    } catch (err) {
      console.error("Microphone access denied:", err);
      alert("Please allow microphone access to initialize the hazard detection system.");
    }
  };

  const stop = () => {
    if (streamRef.current) streamRef.current.getTracks().forEach(track => track.stop());
    if (ctxRef.current && ctxRef.current.state !== "closed") ctxRef.current.close();
    setActive(false);
  };

  return { 
    active, 
    start, 
    stop, 
    source: sourceRef.current, 
    audioContext: ctxRef.current 
  };
}