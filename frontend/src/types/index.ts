export interface Document {
  id: string;
  title: string;
  doc_type: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  created_at: string;
  updated_at: string;
  chunk_count: number;
  metadata: {
    page_count?: number;
    word_count?: number;
    file_size?: number;
    author?: string;
  };
  error_message?: string;
}

export type AnswerMode = 'knowledge_base' | 'external_graph' | 'llm_direct' | 'system';

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  /** From POST /api/chat: how the assistant reply was produced */
  answerMode?: AnswerMode;
  contextUsed?: number;
}

export interface Source {
  chunk_id: string;
  score: number;
  text: string;
  document_id?: string;
  document_title?: string;
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
