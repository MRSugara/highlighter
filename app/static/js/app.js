document.addEventListener('DOMContentLoaded',()=>{
  const form=document.querySelector('form');
  const liveSec=document.getElementById('live-sections');
  const logsList=document.getElementById('logs-list');
  const output=document.getElementById('live-output');

  async function submitFeedbackForm(feedbackForm){
    const clipId=feedbackForm.getAttribute('data-clip-id')||'';
    const statusEl=feedbackForm.querySelector('.feedback-status');
    const btn=feedbackForm.querySelector('button[type="submit"]');
    if(btn){btn.disabled=true;}
    if(statusEl){statusEl.textContent='Menyimpan...';}

    const ratingEl=feedbackForm.querySelector('select[name="rating"]');
    const rating=ratingEl ? Number(ratingEl.value) : null;

    const weaknesses=[...feedbackForm.querySelectorAll('input[name="weaknesses"]:checked')].map(x=>x.value);
    const strengths=[...feedbackForm.querySelectorAll('input[name="strengths"]:checked')].map(x=>x.value);
    const notesEl=feedbackForm.querySelector('textarea[name="notes"]');
    const notes=notesEl ? String(notesEl.value||'').trim() : '';

    try{
      const res=await fetch('/feedback_api',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({
          clip_id: clipId,
          rating: Number.isFinite(rating) ? Math.trunc(rating) : rating,
          weaknesses,
          strengths,
          notes: notes || null,
        })
      });
      const data=await res.json().catch(()=>({}));
      if(!res.ok){
        const msg=(data && data.detail) ? data.detail : ('HTTP '+res.status);
        if(statusEl){statusEl.textContent='Gagal: '+msg;}
      }else{
        if(statusEl){statusEl.textContent='Tersimpan';}
      }
    }catch(err){
      if(statusEl){statusEl.textContent='Gagal: '+err;}
    }finally{
      if(btn){btn.disabled=false;}
    }
  }

  // Delegate feedback submission (works for server-rendered + live-inserted forms)
  document.addEventListener('submit', (e)=>{
    const target=e.target;
    if(!(target instanceof HTMLFormElement)) return;
    if(!target.classList.contains('feedback-form')) return;
    e.preventDefault();
    submitFeedbackForm(target);
  });

  function addLog(msg){
    if(!logsList) return;
    const li=document.createElement('li');
    li.textContent=msg;
    logsList.appendChild(li);
  }

  async function runAnalysis(e){
    e.preventDefault();
    if(!form) return;
    const btn=form.querySelector('button[type="submit"]');
    if(btn){btn.disabled=true; btn.textContent='Analyzing…';}

    if(liveSec){liveSec.style.display='block';}
    if(logsList){logsList.innerHTML='';}
    if(output){output.innerHTML='';}

    addLog('Starting live analysis...');
    const urlInput=form.querySelector('#youtube_url');
    const urlVal=encodeURIComponent(urlInput ? urlInput.value : '');
    const es=new EventSource(`/analyze_stream?youtube_url=${urlVal}`);

    es.addEventListener('log', (ev)=>{
      addLog(ev.data);
    });

    es.addEventListener('result', (ev)=>{
      try{
        const data=JSON.parse(ev.data);
        if(data.error){
          const div=document.createElement('div');
          div.className='alert error';
          div.textContent='Error: '+data.error;
          output.appendChild(div);
        }
        if(data.video_id){
          const embed=document.createElement('div');
          embed.className='embed';
          embed.innerHTML=`<iframe src="https://www.youtube.com/embed/${data.video_id}" title="YouTube video" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>`;
          output.appendChild(embed);
        }
        if(Array.isArray(data.highlights) && data.highlights.length){
          const h=document.createElement('div');
          h.innerHTML='<h4>Highlights</h4>';
          const ul=document.createElement('ul');
          ul.className='highlights';
          data.highlights.forEach(x=>{
            const li=document.createElement('li');
            const t=document.createElement('div');
            t.innerHTML=`<strong>${x.start.toFixed(1)}s → ${x.end.toFixed(1)}s</strong> <span class="badge">Score: ${Number(x.score).toFixed(2)}</span>`;
            li.appendChild(t);

            if(data.video_id){
              const startSec=Math.max(0, Math.floor(Number(x.start)||0));
              const endSec=Math.max(startSec+2, Math.ceil(Number(x.end)||0));
              const embed=document.createElement('div');
              embed.className='embed';
              embed.style.margin='.75rem 0';
              embed.innerHTML=`<iframe src="https://www.youtube.com/embed/${data.video_id}?start=${startSec}&end=${endSec}" title="Highlight at ${x.start.toFixed(1)}s" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>`;
              li.appendChild(embed);
            }

            if(x.category){
              const m=document.createElement('div');
              m.className='muted';
              m.textContent='Category: '+x.category;
              li.appendChild(m);
            }
            if(x.reason){
              const r=document.createElement('div');
              r.textContent=x.reason;
              li.appendChild(r);
            }
            if(x.transcript){
              const b=document.createElement('blockquote');
              b.textContent=x.transcript;
              li.appendChild(b);
            }

            // Feedback UI
            const clipId = x.clip_id || (data.video_id ? `${data.video_id}:${Number(x.start||0).toFixed(1)}-${Number(x.end||0).toFixed(1)}` : `${Number(x.start||0).toFixed(1)}-${Number(x.end||0).toFixed(1)}`);
            const details=document.createElement('details');
            details.style.marginTop='.75rem';
            details.innerHTML=`
              <summary><strong>Beri feedback (untuk belajar)</strong></summary>
              <form class="feedback-form" data-clip-id="${clipId}">
                <div style="margin-top:.5rem;">
                  <label>Rating (1–10)
                    <select name="rating" required>
                      ${Array.from({length:10},(_,i)=>i+1).map(r=>`<option value="${r}">${r}</option>`).join('')}
                    </select>
                  </label>
                </div>
                <div style="margin-top:.5rem;">
                  <div class="muted">Weakness</div>
                  <div class="feedback-grid">
                    ${['too_short','too_long','weak_opening','weak_closing','unclear_topic','no_insight','missing_reasoning','no_takeaway','too_much_context'].map(w=>`<label><input type="checkbox" name="weaknesses" value="${w}"> ${w}</label>`).join('')}
                  </div>
                </div>
                <div style="margin-top:.5rem;">
                  <div class="muted">Strength</div>
                  <div class="feedback-grid">
                    ${['strong_opening','clear_insight','good_duration','strong_reasoning','clear_takeaway','standalone_value','good_topic'].map(s=>`<label><input type="checkbox" name="strengths" value="${s}"> ${s}</label>`).join('')}
                  </div>
                </div>
                <div style="margin-top:.5rem;">
                  <label>Notes (opsional)
                    <textarea name="notes" rows="2" placeholder="Catatan bebas (disimpan saja)"></textarea>
                  </label>
                </div>
                <div style="margin-top:.5rem; display:flex; gap:.5rem; align-items:center;">
                  <button type="submit">Kirim feedback</button>
                  <span class="muted feedback-status" aria-live="polite"></span>
                </div>
              </form>
            `;
            li.appendChild(details);
            ul.appendChild(li);
          });
          h.appendChild(ul);
          output.appendChild(h);
        }
      }catch(err){
        addLog('Failed to parse result: '+err);
      }
      es.close();
      if(btn){btn.disabled=false; btn.textContent='Analyze';}
    });

    es.onerror=(e)=>{
      addLog('Stream error or closed');
      es.close();
      if(btn){btn.disabled=false; btn.textContent='Analyze';}
    };
  }

  if(form){
    form.addEventListener('submit',runAnalysis);
  }
});