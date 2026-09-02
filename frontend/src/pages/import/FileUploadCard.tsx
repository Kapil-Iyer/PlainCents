import { useRef, useState } from "react";
import { Loader2, UploadCloud } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface FileUploadCardProps {
  onUpload: (file: File) => void;
  pending: boolean;
}

export function FileUploadCard({ onUpload, pending }: FileUploadCardProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);
  const [selected, setSelected] = useState<File | null>(null);

  const handleFile = (file: File | undefined) => {
    if (!file) return;
    setSelected(file);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Import a TD CSV export</CardTitle>
        <CardDescription>
          Upload your TD chequing/credit card CSV export. You'll see a preview before anything is
          saved.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div
          className={cn(
            "flex flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed border-border px-6 py-10 text-center transition-colors",
            dragActive && "border-primary bg-primary/5",
          )}
          onDragOver={(e) => {
            e.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragActive(false);
            handleFile(e.dataTransfer.files?.[0]);
          }}
        >
          <UploadCloud className="h-8 w-8 text-muted-foreground" />
          <div>
            <p className="text-sm font-medium">
              {selected ? selected.name : "Drag & drop your CSV here"}
            </p>
            <p className="text-xs text-muted-foreground">or</p>
          </div>
          <Button type="button" variant="outline" size="sm" onClick={() => inputRef.current?.click()}>
            Choose file
          </Button>
          <input
            ref={inputRef}
            type="file"
            accept=".csv,text/csv"
            className="hidden"
            onChange={(e) => handleFile(e.target.files?.[0])}
          />
        </div>

        <div className="mt-4 flex justify-end">
          <Button disabled={!selected || pending} onClick={() => selected && onUpload(selected)}>
            {pending && <Loader2 className="h-4 w-4 animate-spin" />}
            Upload &amp; preview
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
