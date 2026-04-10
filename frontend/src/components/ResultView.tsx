import { ShieldCheck, ShieldAlert, Mic, Video, BarChart3 } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface ResultViewProps {
  result: 'real' | 'fake';
  audioScore: number;
  videoScore: number;
  finalScore: number;
  onReset: () => void;
}

const ResultView = ({ result, audioScore, videoScore, finalScore, onReset }: ResultViewProps) => {
  const isReal = result === 'real';
  const confidence = isReal ? (1 - finalScore) * 100 : finalScore * 100;

  return (
    <div className="max-w-2xl mx-auto space-y-8 py-8">

      {/* RESULT */}
      <div className={`text-center p-6 rounded-xl ${
        isReal ? 'bg-green-50' : 'bg-red-50'
      }`}>
        <div className="flex justify-center mb-4">
          {isReal ? (
            <ShieldCheck className="w-20 h-20 text-green-600" />
          ) : (
            <ShieldAlert className="w-20 h-20 text-red-600" />
          )}
        </div>

        <h2 className="text-4xl font-bold mb-2 text-gray-900">
          {isReal ? 'REAL' : 'FAKE'}
        </h2>

        <p className="text-lg text-gray-600">
          Confidence: {confidence.toFixed(1)}%
        </p>
      </div>

      {/* SCORES */}
      <div className="p-6 bg-white border border-gray-200 rounded-xl shadow-sm">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-blue-600" />
          Model Scores
        </h3>

        <div className="space-y-4">

          {/* AUDIO */}
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 rounded-lg bg-blue-100 flex items-center justify-center">
              <Mic className="w-5 h-5 text-blue-600" />
            </div>

            <div className="flex-1">
              <div className="flex justify-between mb-1">
                <span className="text-sm text-gray-500">Audio Model (CNN)</span>
                <span className="text-sm font-mono text-gray-900">
                  {(audioScore * 100).toFixed(1)}% fake
                </span>
              </div>

              <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500 transition-all duration-500"
                  style={{ width: `${audioScore * 100}%` }}
                />
              </div>
            </div>
          </div>

          {/* VIDEO */}
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 rounded-lg bg-blue-100 flex items-center justify-center">
              <Video className="w-5 h-5 text-blue-600" />
            </div>

            <div className="flex-1">
              <div className="flex justify-between mb-1">
                <span className="text-sm text-gray-500">Video Model (ResNet18)</span>
                <span className="text-sm font-mono text-gray-900">
                  {(videoScore * 100).toFixed(1)}% fake
                </span>
              </div>

              <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500 transition-all duration-500"
                  style={{ width: `${videoScore * 100}%` }}
                />
              </div>
            </div>
          </div>

          {/* FUSION */}
          <div className="pt-4 border-t border-gray-200">
            <div className="flex items-center justify-between">
              <span className="text-gray-900 font-medium">
                Final Score (Weighted Fusion)
              </span>
              <span className="text-xl font-mono font-bold text-blue-600">
                {(finalScore * 100).toFixed(1)}%
              </span>
            </div>

            <p className="text-xs text-gray-500 mt-1">
              Formula: 0.2 × Audio + 0.8 × Video
            </p>
          </div>

        </div>
      </div>

      {/* BUTTON */}
      <Button onClick={onReset} className="w-full" size="lg">
        Analyze Another Video
      </Button>

    </div>
  );
};

export default ResultView;