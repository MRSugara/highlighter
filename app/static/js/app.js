document.addEventListener('DOMContentLoaded',()=>{
  const form=document.querySelector('form');
  const liveSec=document.getElementById('live-sections');
  const logsList=document.getElementById('logs-list');
  const output=document.getElementById('live-output');

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