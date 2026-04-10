import { useCallback } from 'react';
import { Upload, Film } from 'lucide-react';

interface VideoUploadProps {
  onVideoSelect: (file: File) => void;
  selectedVideo: File | null;
}

const VideoUpload = ({ onVideoSelect, selectedVideo }: VideoUploadProps) => {
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('video/')) {
      onVideoSelect(file);
    }
  }, [onVideoSelect]);

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onVideoSelect(file);
    }
  }, [onVideoSelect]);

  return (
    <div
      onDrop={handleDrop}
      onDragOver={(e) => e.preventDefault()}
      className="glass-card p-8 border-2 border-dashed border-border hover:border-primary/50 transition-all duration-300 cursor-pointer group"
    >
      <label className="flex flex-col items-center justify-center gap-4 cursor-pointer">
        <input
          type="file"
          accept="video/*"
          onChange={handleChange}
          className="hidden"
        />
        
        {selectedVideo ? (
          <>
            <div className="w-16 h-16 rounded-full bg-primary/20 flex items-center justify-center">
              <Film className="w-8 h-8 text-primary" />
            </div>
            <div className="text-center">
              <p className="text-foreground font-medium">{selectedVideo.name}</p>
              <p className="text-muted-foreground text-sm">
                {(selectedVideo.size / (1024 * 1024)).toFixed(2)} MB
              </p>
            </div>
            <p className="text-primary text-sm">Click or drag to replace</p>
          </>
        ) : (
          <>
            <div className="w-16 h-16 rounded-full bg-secondary flex items-center justify-center group-hover:bg-primary/20 transition-colors">
              <Upload className="w-8 h-8 text-muted-foreground group-hover:text-primary transition-colors" />
            </div>
            <div className="text-center">
              <p className="text-foreground font-medium">Drop your video here</p>
              <p className="text-muted-foreground text-sm">or click to browse</p>
            </div>
            <p className="text-muted-foreground text-xs">
              Supports MP4, AVI, MOV, MKV
            </p>
          </>
        )}
      </label>
    </div>
  );
};

export default VideoUpload;
