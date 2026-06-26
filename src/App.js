import React, { useEffect, useState, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import Meyda from "meyda";
import { useMicrophone } from "./Microphone";

const LABELS = ["Ambulance", "Horn", "Train Horn", "Safe Background"];

export default function App() {
  const [model, setModel] = useState(null);
  const [logs, setLogs] = useState([]);
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [currentClass, setCurrentClass] = useState(3); // 3 = Background
  
  const mic = useMicrophone();
  const analyzerRef = useRef(null);
  const mfccHistory = useRef([]); 

  useEffect(() => {
    tf.loadLayersModel("/tfjs_model/model.json")
      .then((m) => {
        setModel(m);
        console.log("Deep CNN Model loaded successfully!");
        tf.tidy(() => m.predict(tf.zeros([1, 40, 174, 1])));
      })
      .catch((err) => console.error("Failed to load model.", err));
  }, []);

  const runPrediction = (frames) => {
    if (!model) return;

    const { classIdx, confidence } = tf.tidy(() => {
      let tensor2d = tf.tensor2d(frames).transpose(); 
      
      const padAmount = 174 - tensor2d.shape[1];
      if (padAmount > 0) {
        tensor2d = tensor2d.pad([[0, 0], [0, padAmount]]);
      } else {
        tensor2d = tensor2d.slice([0, 0], [40, 174]);
      }

      const mean = tensor2d.mean();
      const variance = tensor2d.sub(mean).square().mean();
      const std = tf.sqrt(variance);
      const normalized = tensor2d.sub(mean).div(std.add(1e-7));

      const input = normalized.reshape([1, 40, 174, 1]);
      const pred = model.predict(input);
      
      return {
        classIdx: pred.argMax(1).dataSync()[0],
        confidence: pred.max().dataSync()[0]
      };
    });

    setCurrentClass(classIdx);

    if (confidence > 0.85 && classIdx < 3) {
      const patterns = { 0: [300, 100, 300], 1: [200, 50, 200], 2: [500, 100, 500] };
      if ("vibrate" in navigator) navigator.vibrate(patterns[classIdx]);

      const newLog = { 
        label: LABELS[classIdx], 
        time: new Date().toLocaleTimeString(), 
        conf: Math.round(confidence * 100) 
      };
      setLogs((prev) => [newLog, ...prev.slice(0, 4)]);
    }
  };

  useEffect(() => {
    if (mic.active && mic.source && mic.audioContext) {
      Meyda.numberOfMFCCCoefficients = 40;
      
      analyzerRef.current = Meyda.createMeydaAnalyzer({
        audioContext: mic.audioContext,
        source: mic.source,
        bufferSize: 512, 
        featureExtractors: ["mfcc"],
        callback: (features) => {
          if (features && features.mfcc) {
            mfccHistory.current.push(features.mfcc);
            if (mfccHistory.current.length >= 32) {
              runPrediction(mfccHistory.current);
              mfccHistory.current = []; 
            }
          }
        }
      });
      analyzerRef.current.start();
    } else {
      if (analyzerRef.current) analyzerRef.current.stop();
      mfccHistory.current = [];
      setCurrentClass(3); 
    }
    
    return () => {
      if (analyzerRef.current) analyzerRef.current.stop();
    };
  }, [mic.active, model]);

  // --- UI DESIGN SYSTEM ---
  
  const theme = {
    bg: isDarkMode ? "#0a0a0a" : "#f1f3f5", // Deepened body backgrounds to maximize contrast
    card: isDarkMode ? "#161616" : "#ffffff",
    text: isDarkMode ? "#ffffff" : "#1a1a1a",
    textMuted: isDarkMode ? "#888888" : "#666666",
    border: isDarkMode ? "#2c2c2c" : "#e0e0e0",
    buttonStart: isDarkMode ? "#4CAF50" : "#2e7d32",
    buttonStop: isDarkMode ? "#f44336" : "#c62828",
  };

  // Heavy, deeply saturated dark colors with solid 75% opacity targets
  const getPulseColor = () => {
    switch (currentClass) {
      case 0: return "rgba(139, 0, 0, 0.75)";       // Dark Blood Crimson (Ambulance)
      case 1: return "rgba(179, 134, 0, 0.80)";      // Dark Saturated Amber (Horn)
      case 2: return "rgba(10, 37, 102, 0.75)";      // Midnight Navy Blue (Train Horn)
      default: return "rgba(0, 0, 0, 0)";
    }
  };

  return (
    <div style={{ 
      minHeight: "100vh", 
      backgroundColor: theme.bg, 
      color: theme.text,
      transition: "background-color 0.4s ease, color 0.3s ease", 
      padding: "20px", 
      fontFamily: "system-ui, -apple-system, sans-serif",
      position: "relative",
      overflow: "hidden" 
    }}>
      
      {/* Enhanced Animation Profile */}
      <style>
        {`
          @keyframes radarPulse {
            0% { transform: translate(-50%, -50%) scale(0.75); opacity: 0.9; }
            50% { transform: translate(-50%, -50%) scale(1.05); opacity: 0.45; }
            100% { transform: translate(-50%, -50%) scale(0.75); opacity: 0.9; }
          }
        `}
      </style>

      {/* High-Opaqueness Extended-Radius Background Pulse */}
      <div style={{
        position: "absolute",
        top: "50%",
        left: "50%",
        width: "1400px", // Expanded radius
        height: "1400px", // Expanded radius
        borderRadius: "50%",
        pointerEvents: "none", 
        // Gradient structure transitions through secondary fallback points before true transparency
        background: `radial-gradient(circle, ${getPulseColor()} 0%, ${getPulseColor().replace("0.75", "0.3").replace("0.80", "0.3")} 45%, rgba(0,0,0,0) 75%)`,
        animation: currentClass < 3 ? "radarPulse 2.4s infinite ease-in-out" : "none",
        opacity: currentClass < 3 ? 1 : 0,
        transition: "opacity 0.4s ease, background 0.4s ease",
        zIndex: 0
      }} />

      {/* Main Interface Content Layer */}
      <div style={{ position: "relative", zIndex: 1 }}>
        
        {/* Top Navigation Bar */}
        <div style={{ maxWidth: "500px", margin: "0 auto", display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "30px" }}>
          <div>
            <h1 style={{ margin: "0", fontSize: "28px", letterSpacing: "-0.5px" }}>HearSight</h1>
            <p style={{ margin: "4px 0 0 0", color: theme.textMuted, fontSize: "14px", fontWeight: "500" }}>Acoustic Hazard Engine</p>
          </div>
          
          <button 
            onClick={() => setIsDarkMode(!isDarkMode)}
            style={{
              background: theme.card,
              border: `2px solid ${theme.border}`,
              color: theme.text,
              padding: "8px 16px",
              borderRadius: "20px",
              cursor: "pointer",
              fontWeight: "bold",
              transition: "all 0.2s"
            }}
          >
            {isDarkMode ? "☀️ Day" : "🌙 Night"}
          </button>
        </div>

        {/* Main Dashboard Card */}
        <div style={{ 
          maxWidth: "500px", 
          margin: "0 auto", 
          backgroundColor: theme.card,
          border: `2px solid ${theme.border}`,
          borderRadius: "16px",
          padding: "24px",
          transition: "background-color 0.3s, border-color 0.3s"
        }}>
          
          <div style={{ 
            padding: "12px 16px", 
            backgroundColor: model ? (isDarkMode ? "#1b5e2022" : "#e8f5e9") : (isDarkMode ? "#fff8e122" : "#fff8e1"), 
            border: `2px solid ${model ? "#4CAF50" : "#ffb300"}`,
            borderRadius: "8px", 
            marginBottom: "24px",
            fontWeight: "600",
            display: "flex",
            alignItems: "center",
            gap: "10px"
          }}>
            {model ? "✅ Neural Network Online" : "⏳ Booting Core Sequence..."}
          </div>
          
          <button 
            onClick={mic.active ? mic.stop : mic.start}
            disabled={!model}
            style={{
              width: "100%", padding: "18px", fontSize: "18px",
              backgroundColor: mic.active ? theme.buttonStop : theme.buttonStart,
              color: "#ffffff", 
              border: "none", 
              borderRadius: "12px", 
              cursor: model ? "pointer" : "not-allowed",
              fontWeight: "700", 
              letterSpacing: "0.5px",
              transition: "background-color 0.2s, transform 0.1s"
            }}
            onMouseDown={(e) => e.currentTarget.style.transform = "scale(0.98)"}
            onMouseUp={(e) => e.currentTarget.style.transform = "scale(1)"}
            onMouseLeave={(e) => e.currentTarget.style.transform = "scale(1)"}
          >
            {mic.active ? "STOP SCANNER" : "ACTIVATE SCANNER"}
          </button>

          <h3 style={{ marginTop: "40px", marginBottom: "16px", fontSize: "16px", color: theme.textMuted, textTransform: "uppercase", letterSpacing: "1px" }}>
            Incident Log
          </h3>
          
          <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
            {logs.length === 0 ? (
              <div style={{ 
                padding: "40px 20px", 
                textAlign: "center", 
                border: `2px dashed ${theme.border}`, 
                borderRadius: "12px",
                color: theme.textMuted,
                fontWeight: "500"
              }}>
                Monitoring environment...
              </div>
            ) : null}
            
            {logs.map((log, i) => (
              <div key={i} style={{ 
                border: `2px solid ${theme.border}`, 
                padding: "16px", 
                borderRadius: "12px",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                backgroundColor: isDarkMode ? "#2a2a2a" : "#fafafa"
              }}>
                <div>
                  <div style={{ fontSize: "16px", fontWeight: "bold" }}>🚨 {log.label}</div>
                  <div style={{ fontSize: "12px", color: theme.textMuted, marginTop: "6px", fontWeight: "500" }}>{log.time}</div>
                </div>
                <div style={{ 
                  backgroundColor: isDarkMode ? "#332a00" : "#fff8e1", 
                  border: "2px solid #ffb300",
                  color: isDarkMode ? "#ffd54f" : "#f57f17",
                  padding: "6px 12px", 
                  borderRadius: "20px", 
                  fontSize: "14px", 
                  fontWeight: "bold" 
                }}>
                  {log.conf}%
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}