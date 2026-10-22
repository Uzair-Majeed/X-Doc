const API = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
export async function get<T>(path: string): Promise<T> {
  const response = await fetch(`${API}${path}`)
  const body = await response.json()
  if (!response.ok) throw new Error(body.detail || 'Could not connect to the local backend.')
  return body as T
}
export async function postImage(file: File, keys: string[]) {
  const form = new FormData(); form.append('image', file); form.append('model_keys', keys.join(','))
  const response = await fetch(`${API}/api/predict`, { method: 'POST', body: form }); const body = await response.json()
  if (!response.ok) throw new Error(body.detail || 'Prediction could not be completed.')
  return body as {results: Prediction[]; notice: string}
}
export async function clearHistory() {
  const response = await fetch(`${API}/api/predictions`, { method: 'DELETE' }); const body = await response.json()
  if (!response.ok) throw new Error(body.detail || 'History could not be cleared.')
}
export type Model = {key:string;name:string;file:string;available:boolean;status:string;error?:string|null;metrics?:{accuracy:string|null;roc_auc:string|null;macro_f1:string|null}}
export type Prediction = {model_key:string;model_name:string;predicted_label:string;scores:Record<string,number>;inference_ms:number;input_size:string}
export type History = {total:number;rows:{id:number;created_at:string;model_key:string;original_filename:string;predicted_label:string;inference_ms:number}[]}
