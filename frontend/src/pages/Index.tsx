import { useState, useCallback } from 'react';
import { ScanSearch, Github, BookOpen } from 'lucide-react';
import { Button } from '@/components/ui/button';
import VideoUpload from '@/components/VideoUpload';
import ProcessingView from '@/components/ProcessingView';
import ResultView from '@/components/ResultView';

type AppState = 'idle' | 'processing' | 'result';
type ProcessingStage = 'extracting' | 'audio' | 'video' | 'fusion';

interface AnalysisResult {
  result: 'real' | 'fake';
  audioScore: number;
  videoScore: number;
  finalScore: number;
}

const Index = () => {
  const [state, setState] = useState<AppState>('idle');
  const [selectedVideo, setSelectedVideo] = useState<File | null>(null);
  const [processingStage, setProcessingStage] = useState<ProcessingStage>('extracting');
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  // 🔥 REAL BACKEND ANALYSIS FUNCTION
  const analyzeWithBackend = useCallback(async () => {
    if (!selectedVideo) {
      alert("Please select a video first.");
      return;
    }

    setState('processing');
    setProgress(10);
    setProcessingStage('extracting');

    const formData = new FormData();
    formData.append("file", selectedVideo);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Backend error");
      }

      setProcessingStage('audio');
      setProgress(40);

      const data = await response.json();

      setProcessingStage('video');
      setProgress(70);

      setProcessingStage('fusion');
      setProgress(100);

      setResult({
        result: data.prediction === "FAKE" ? 'fake' : 'real',
        audioScore: data.audio_score,
        videoScore: data.video_score,
        finalScore: data.final_score,
      });

      setState('result');

    } catch (error) {
      console.error("Backend error:", error);
      alert("Backend connection failed. Make sure backend is running.");
      setState('idle');
    }
  }, [selectedVideo]);

  const handleReset = useCallback(() => {
    setState('idle');
    setSelectedVideo(null);
    setResult(null);
    setProgress(0);
    setProcessingStage('extracting');
  }, []);

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-primary to-accent flex items-center justify-center">
                <ScanSearch className="w-6 h-6 text-primary-foreground" />
              </div>
              <div>
                <h1 className="text-xl font-bold gradient-text">Truth-Seeker</h1>
                <p className="text-xs text-muted-foreground">Multimodal Deepfake Detection</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Button variant="ghost" size="icon">
                <BookOpen className="w-5 h-5" />
              </Button>
              <Button variant="ghost" size="icon">
                <Github className="w-5 h-5" />
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="container mx-auto px-4 py-8 max-w-2xl">

        {state === 'idle' && (
          <>
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-foreground mb-3">
                Detect Deepfake Videos
              </h2>
              <p className="text-muted-foreground max-w-md mx-auto">
                Upload a video to analyze using our multimodal AI pipeline.
              </p>
            </div>

            <VideoUpload
              onVideoSelect={setSelectedVideo}
              selectedVideo={selectedVideo}
            />

            {selectedVideo && (
              <Button
                onClick={analyzeWithBackend}
                className="w-full mt-6"
                size="lg"
              >
                <ScanSearch className="w-5 h-5 mr-2" />
                Detect Deepfake
              </Button>
            )}
          </>
        )}

        {state === 'processing' && (
          <ProcessingView stage={processingStage} progress={progress} />
        )}

        {state === 'result' && result && (
          <ResultView
            result={result.result}
            audioScore={result.audioScore}
            videoScore={result.videoScore}
            finalScore={result.finalScore}
            onReset={handleReset}
          />
        )}

      </main>

      {/* Footer */}
      <footer className="border-t border-border/50 mt-auto">
        <div className="container mx-auto px-4 py-6 text-center">
          <p className="text-sm text-muted-foreground">
            MCA Major Project • Multimodal Deepfake Detection
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Index;