export interface Document {
  id: string;
  title: string;
  doc_type: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  created_at: string;
  updated_at: string;
  chunk_count: number;
  metadata: {
    word_count?: number;
    file_size?: number;
    author?: string;
  };
  error_message?: string;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
}

export interface Source {
  chunk_id: string;
  score: number;
  text: string;
}

export interface Stats {
  documents: {
    total: number;
    by_status: Record<string, number>;
  };
  chunks: {
    total: number;
  };
}
