import axios from 'axios';

/** Long timeout: first embedding model load / document processing can take many minutes. */
const LONG_MS = 15 * 60 * 1000;

// Do not set default Content-Type: application/json — with FormData uploads, axios would
// stringify the body as JSON (see defaults transformRequest) and FastAPI returns 422.
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  timeout: 120_000,
});

export default api;

export const fetchDocuments = async () => {
  const response = await api.get('/api/documents/');
  return response.data;
};

export const fetchStats = async () => {
  const response = await api.get('/api/stats');
  return response.data;
};

export const uploadDocument = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);

  // Do not set Content-Type: browser/axios must set multipart boundary for FormData
  const response = await api.post('/api/documents/', formData, { timeout: LONG_MS });
  return response.data;
};

export const deleteDocument = async (id: string) => {
  await api.delete(`/api/documents/${id}`);
};

export const queryChat = async (message: string, history: any[]) => {
  const response = await api.post(
    '/api/chat',
    {
      message,
      history,
    },
    { timeout: LONG_MS },
  );
  return response.data;
};
