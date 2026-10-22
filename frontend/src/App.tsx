import { useEffect, useMemo, useState, type ChangeEvent, type DragEvent } from 'react'
import { clearHistory, get, postImage, type History, type Model, type Prediction } from './api'

type Page = 'Home' | 'Prediction Studio'
const modelOrder = ['vgg16_fine_tuned', 'vgg16_frozen', 'resnet50_fine_tuned', 'resnet50_frozen', 'custom_cnn']

export default function App() {
  const [page, setPage] = useState<Page>('Home')
  const [models, setModels] = useState<Model[]>([])
  const [history, setHistory] = useState<History | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState('')
  const [selectedModel, setSelectedModel] = useState('')
  const [result, setResult] = useState<Prediction | null>(null)
  const [busy, setBusy] = useState(false)
  const [drag, setDrag] = useState(false)
  const [error, setError] = useState('')
  const [connected, setConnected] = useState(false)

  async function refresh() {
    try {
      const [modelData, records] = await Promise.all([
        get<{models:Model[]}>('/api/models'), get<History>('/api/predictions?limit=20'),
      ])
      const sorted = modelData.models.sort((a,b) => modelOrder.indexOf(a.key) - modelOrder.indexOf(b.key))
      setModels(sorted); setHistory(records); setConnected(true)
      setSelectedModel(current => current && sorted.some(m=>m.key===current&&m.available) ? current : sorted.find(m=>m.available)?.key || '')
    } catch (err) { setConnected(false); setError(err instanceof Error ? err.message : 'Could not reach the local backend.') }
  }
  useEffect(() => { void refresh() }, [])
  useEffect(() => {
    if (!file) { setPreview(''); return }
    const url=URL.createObjectURL(file); setPreview(url); return ()=>URL.revokeObjectURL(url)
  }, [file])

  const available=useMemo(()=>models.filter(m=>m.available),[models])
  function acceptFile(next?:File) {
    setError(''); setResult(null)
    if(!next)return
    if(!['image/png','image/jpeg','image/webp'].includes(next.type)){setError('Choose a PNG, JPEG, or WebP image.');return}
    if(next.size>12*1024*1024){setError('Image exceeds the 12 MB upload limit.');return}
    setFile(next)
  }
  function onDrop(e:DragEvent<HTMLDivElement>){e.preventDefault();setDrag(false);acceptFile(e.dataTransfer.files[0])}
  async function predict(){
    if(!file||!selectedModel)return
    setBusy(true);setError('');setResult(null)
    try{const data=await postImage(file,[selectedModel]);setResult(data.results[0]);setHistory(await get<History>('/api/predictions?limit=20'))}
    catch(err){setError(err instanceof Error?err.message:'Prediction failed.');try{await refresh()}catch{/* current error is enough */}}
    finally{setBusy(false)}
  }
  async function clear(){
    if(!window.confirm('Clear all saved prediction history?'))return
    try{await clearHistory();setHistory(await get<History>('/api/predictions?limit=20'))}
    catch(err){setError(err instanceof Error?err.message:'History could not be cleared.')}
  }

  return <div className="app-shell">
    <header className="topbar"><button className="brand" onClick={()=>setPage('Home')} aria-label="X-Doc home"><span className="brand-mark">X</span><span>X-Doc</span></button><div className="top-meta"><span className={`local-dot ${connected?'':'offline'}`}/>{connected?'Local backend connected':'Backend not connected'}<span className="divider"/>CNN workspace</div><span className="avatar">XD</span></header>
    <div className="layout"><aside className="sidebar"><div className="side-label">WORKSPACE</div><nav>{(['Home','Prediction Studio'] as Page[]).map((p,i)=><button key={p} onClick={()=>{setPage(p);setError('')}} className={`nav-item ${page===p?'active':''}`}><span className="nav-icon">{i?'⌕':'⌂'}</span>{p}</button>)}</nav><div className="side-bottom"><div className="side-label">MODEL FILES</div><div className="mini-model"><span className="status-dot"/><div><b>{available.length} detected</b><small>loaded when selected</small></div></div><div className="disclaimer"><span>ⓘ</span><p><b>Educational use only</b><br/>Model outputs are not a medical diagnosis.</p></div></div></aside>
      <main className="main-content"><div className="page-heading"><div><div className="eyebrow">X-DOC <span>/</span> LOCAL CNN WORKSPACE</div><h1>{page==='Home'?'Explore your models':page}</h1><p>{page==='Home'?'Run and inspect your chest X-ray CNN models in a local workspace.':'Choose a model, upload an image, and inspect its prediction.'}</p></div>{page==='Prediction Studio'&&<span className="status-pill"><i/> Local inference</span>}</div>
        {error&&<div className="alert" role="alert"><span>!</span>{error}<button onClick={()=>setError('')} aria-label="Dismiss">×</button></div>}
        {page==='Home'?<Home models={models} history={history} go={setPage} clear={clear}/>:<section className="studio-grid">
          <div className="panel upload-panel"><div className="panel-head"><div><h2>Prediction setup</h2><p>Choose a model and upload an image.</p></div><span className="step">01</span></div>
            <div className="model-select"><div className="select-head"><h3>Choose model</h3><span>Sorted by your supplied results</span></div>{available.length?<details className="model-picker"><summary className="picker-trigger"><span className="picker-model">{available.find(m=>m.key===selectedModel)?.name||'Select a model'}<small>{available.find(m=>m.key===selectedModel)?.file||'Local CNN'}</small></span><span className="picker-metrics"><i>Acc {percent(available.find(m=>m.key===selectedModel)?.metrics?.accuracy)}</i><i>AUC {percent(available.find(m=>m.key===selectedModel)?.metrics?.roc_auc)}</i><i>F1 {percent(available.find(m=>m.key===selectedModel)?.metrics?.macro_f1)}</i></span><b className="picker-chevron">⌄</b></summary><div className="picker-menu" role="radiogroup" aria-label="Select a model">{available.map(m=><label key={m.key} className={`model-option ${selectedModel===m.key?'chosen':''}`}><input type="radio" name="model" value={m.key} checked={selectedModel===m.key} onChange={e=>{setSelectedModel(m.key);setResult(null);e.currentTarget.closest('details')?.removeAttribute('open')}}/><span className="radio-indicator"/><span className="model-text"><b>{m.name}</b><small>{m.file}</small><span className="model-metrics"><i>Acc {percent(m.metrics?.accuracy)}</i><i>AUC {percent(m.metrics?.roc_auc)}</i><i>F1 {percent(m.metrics?.macro_f1)}</i></span></span></label>)}</div></details>:<div className="empty-inline">No model files detected. Place the supplied .keras files in the models folder and check TensorFlow setup.</div>}</div>
            <div className={`dropzone ${drag?'dragging':''} ${preview?'has-preview':''}`} onDragOver={e=>{e.preventDefault();setDrag(true)}} onDragLeave={()=>setDrag(false)} onDrop={onDrop}>
              {preview?<><img className="preview-image" src={preview} alt="Selected chest X-ray preview"/><div className="preview-meta"><b>{file?.name}</b><span>{file?`${(file.size/1024/1024).toFixed(2)} MB`:''}</span><button className="text-button" onClick={()=>{setFile(null);setResult(null)}}>Remove</button></div></>:<><div className="upload-icon">↑</div><b>Drop an X-ray image here</b><span>or browse from your device</span><label className="button secondary browse">Browse files<input type="file" accept="image/png,image/jpeg,image/webp" onChange={(e:ChangeEvent<HTMLInputElement>)=>acceptFile(e.target.files?.[0])}/></label><small>PNG, JPEG or WebP · Max 12 MB</small></>}
            </div>
            <button className="button primary run-button" disabled={!file||!selectedModel||busy} onClick={predict}>{busy?<><span className="spinner"/>Running inference…</>:'Run prediction'}</button><p className="privacy-note">Image bytes are processed locally and aren’t saved to prediction history.</p>
          </div>
          <div className="panel results-panel"><div className="panel-head"><div><h2>Model output</h2><p>Score, selected model, and visual attribution if available.</p></div><span className="step">02</span></div>
            {busy?<div className="loading-state"><span className="spinner large"/><b>Running inference</b><span>Loading the selected CNN and processing your image locally…</span></div>:result?<ResultView result={result}/>:<div className="empty-results"><div className="empty-art"><span>⌕</span><i/></div><b>No prediction yet</b><p>Select a model and image, then run inference to see the model output here.</p></div>}
            <div className="notice"><span>ⓘ</span><p><b>Experimental output</b><br/>CNN scores are not calibrated patient probabilities and are not a diagnosis.</p></div>
          </div>
        </section>}
        <footer><span>X-Doc <i>·</i> Local research workspace</span><span>Model predictions are not medical diagnoses.</span></footer>
      </main>
    </div>
  </div>
}

function percent(value:string|null|undefined){if(value===null||value===undefined||value==='')return '—';const n=Number(value);return Number.isFinite(n)?`${(n*100).toFixed(2)}%`:'—'}
function Home({models,history,go,clear}:{models:Model[];history:History|null;go:(p:Page)=>void;clear:()=>void}){
  return <><section className="hero panel"><div className="hero-copy"><span className="hero-tag"><i/> LOCAL CNN WORKSPACE</span><h2>Explore model outputs.<br/><em>Stay curious.</em></h2><p>Run your trained chest X-ray CNNs, inspect their scores, and see which image regions influenced an output. Everything runs on your machine.</p><div className="hero-actions"><button className="button primary" onClick={()=>go('Prediction Studio')}>Open Prediction Studio <span>→</span></button></div><div className="hero-disclaimer">ⓘ &nbsp;For educational experimentation only. Not intended for medical use.</div></div><div className="hero-graphic"><div className="orbit orbit-one"/><div className="orbit orbit-two"/><div className="scan-card"><div className="scan-top"><span>X-DOC · LOCAL</span><i/></div><div className="xray"><span className="lung left"/><span className="lung right"/><span className="spine"/><span className="ribs"/></div><div className="scan-bottom"><span>224 × 224 <small>INPUT</small></span><span className="scan-line"/><span>CNN <small>MODEL</small></span></div></div><div className="floating-chip chip-top"><i/> {models.length} model files</div><div className="floating-chip chip-bottom"><span>⌁</span> Local &amp; private</div></div></section>
    <section className="home-bottom"><div className="panel quick-panel"><div className="panel-head"><div><h2>Choose a model</h2><p>Five saved CNNs, sorted by the results you provided.</p></div></div><div className="quick-steps model-summary">{models.map(m=><div key={m.key}><i>{m.name.startsWith('VGG')?'V':'R'}</i><span><b>{m.name}</b><small>{m.file} · Acc {percent(m.metrics?.accuracy)} · AUC {percent(m.metrics?.roc_auc)} · F1 {percent(m.metrics?.macro_f1)}</small></span></div>)}</div><button className="button secondary home-cta" onClick={()=>go('Prediction Studio')}>Select a model <span>→</span></button></div><div className="panel models-home"><div className="panel-head"><div><h2>Recent predictions</h2><p>Metadata saved locally on this device.</p></div>{history?.total?<button className="text-button muted-button" onClick={clear}>Clear history</button>:null}</div><HistoryTable history={history}/></div></section>
  </>
}
function ResultView({result:r}:{result:Prediction}){
  const pneumonia=r.scores.PNEUMONIA??0
  const favorPneumonia=r.predicted_label==='PNEUMONIA'
  return <div className="result-view"><div className="result-card"><div className="result-top"><div className="result-name"><span className="model-symbol">{r.model_name.startsWith('VGG')?'V':'R'}</span><div><b>{r.model_name}</b><small>{r.input_size}</small></div></div><span className={`prediction-label ${!favorPneumonia?'normal':''}`}>{favorPneumonia?'PNEUMONIA':'NORMAL'}</span></div><div className="scores"><div className="score-head"><span>MODEL-REPORTED SCORES</span><span>{r.inference_ms.toFixed(1)} ms</span></div>{Object.entries(r.scores).map(([name,value])=><div className="score-row" key={name}><span>{name}</span><div className="meter"><i style={{width:`${value*100}%`}}/></div><b>{(value*100).toFixed(2)}%</b></div>)}</div></div>
    <div className="output-reason"><b>How to read this output</b><p>{favorPneumonia?`The model's pneumonia score is ${(pneumonia*100).toFixed(2)}%, at or above the 50% classification threshold, so the model labels this image PNEUMONIA.`:`The model's pneumonia score is ${(pneumonia*100).toFixed(2)}%, below the 50% classification threshold, so the model labels this image NORMAL.`} This rule-based text describes the model score only; it does not identify image findings or explain clinical causes.</p></div>
  </div>
}
function HistoryTable({history}:{history:History|null}){if(!history?.rows.length)return <div className="history-empty">No saved predictions yet.</div>;return <div className="table-wrap"><table><thead><tr><th>IMAGE FILE</th><th>MODEL</th><th>OUTPUT</th><th>TIME</th></tr></thead><tbody>{history.rows.slice(0,5).map(r=><tr key={r.id}><td>{r.original_filename||'Image'}</td><td>{r.model_key}</td><td><span className={`history-label ${r.predicted_label==='NORMAL'?'normal':''}`}>{r.predicted_label}</span></td><td>{r.inference_ms.toFixed(1)} ms</td></tr>)}</tbody></table></div>}
