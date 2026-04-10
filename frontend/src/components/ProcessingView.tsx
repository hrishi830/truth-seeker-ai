import { Mic, Video, Loader2, Check } from 'lucide-react';
import { Progress } from '@/components/ui/progress';

interface ProcessingViewProps {
  stage: 'extracting' | 'audio' | 'video' | 'fusion' | 'complete';
  progress: number;
}

const stages = [
  { id: 'extracting', label: 'Extracting Audio & Frames', icon: Loader2 },
  { id: 'audio', label: 'Audio Analysis (CNN)', icon: Mic },
  { id: 'video', label: 'Video Analysis (ResNet18)', icon: Video },
  { id: 'fusion', label: 'Late Fusion', icon: Loader2 },
];

const ProcessingView = ({ stage, progress }: ProcessingViewProps) => {
  const currentIndex = stages.findIndex(s => s.id === stage);

  return (
    <div className="max-w-2xl mx-auto py-10 px-4">
      
      {/* Header */}
      <div className="text-center mb-8">
        <h3 className="text-2xl font-semibold text-gray-900 mb-2">
          Analyzing Video
        </h3>
        <p className="text-gray-500 text-sm">
          Running multimodal deepfake detection pipeline
        </p>
      </div>

      {/* Steps */}
      <div className="space-y-3">
        {stages.map((s, index) => {
          const isActive = s.id === stage;
          const isComplete = index < currentIndex;
          const Icon = s.icon;

          return (
            <div
              key={s.id}
              className={`flex items-center gap-4 p-4 rounded-lg transition ${
                isActive
                  ? 'bg-blue-50 border border-blue-200'
                  : isComplete
                  ? 'bg-green-50 border border-green-200'
                  : 'bg-gray-50 border border-gray-200'
              }`}
            >
              {/* Icon */}
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center ${
                  isActive
                    ? 'bg-blue-100'
                    : isComplete
                    ? 'bg-green-100'
                    : 'bg-gray-200'
                }`}
              >
                {isComplete ? (
                  <Check className="w-5 h-5 text-green-600" />
                ) : (
                  <Icon
                    className={`w-5 h-5 ${
                      isActive
                        ? 'text-blue-600 animate-spin'
                        : 'text-gray-500'
                    }`}
                  />
                )}
              </div>

              {/* Text */}
              <div className="flex-1">
                <p
                  className={`font-medium ${
                    isActive
                      ? 'text-blue-700'
                      : isComplete
                      ? 'text-green-700'
                      : 'text-gray-600'
                  }`}
                >
                  {s.label}
                </p>

                {isActive && (
                  <p className="text-xs text-gray-500 mt-1 font-mono">
                    Processing...
                  </p>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Progress Bar */}
      <div className="mt-8">
        <div className="flex justify-between text-sm mb-2">
          <span className="text-gray-500">Progress</span>
          <span className="text-blue-600 font-mono">
            {Math.round(progress)}%
          </span>
        </div>
        <Progress value={progress} className="h-2" />
      </div>
    </div>
  );
};

export default ProcessingView;