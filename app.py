
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Prediction Results</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <link rel="stylesheet" href="{{ url_for('static', filename='prediction.css') }}"/>

  <style>
    .result-container { padding: 16px; }
    .top-bar { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }
    .top-bar h2 { font-size: 18px; font-weight: 600; }
    .icon-btn { width: 36px; height: 36px; border-radius: 10px; border: 1px solid #e5e7eb; display: flex; align-items: center; justify-content: center; font-size: 18px; cursor: pointer; text-decoration: none; color: inherit; }
    .card { border: 2px solid #3b82f6; border-radius: 18px; padding: 16px; margin-bottom: 16px; background: #fff; }
    .probability { text-align: center; }
    .probability h3 { font-size: 14px; color: #2563eb; margin-bottom: 8px; }
    .probability .score { font-size: 56px; font-weight: 800; color: #60a5fa; }
    .probability p { font-size: 13px; color: #6b7280; }
    .section-title { font-size: 15px; font-weight: 600; margin: 20px 0 10px; }
    .factor { margin-bottom: 12px; }
    .factor span { font-size: 13px; display: block; margin-bottom: 6px; }
    .bar { height: 10px; background: #e5e7eb; border-radius: 999px; overflow: hidden; }
    .bar-fill { height: 100%; background: #60a5fa; border-radius: 999px; }
    .dual-cards { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 10px; }
    .mini-card { border: 2px solid #3b82f6; border-radius: 18px; padding: 14px; }
    .mini-card h4 { font-size: 14px; font-weight: 600; margin-bottom: 6px; }
    .mini-card .big { font-size: 24px; font-weight: 700; color: #60a5fa; }
    .mini-card p { font-size: 12px; color: #6b7280; }
    .disclaimer { font-size: 11px; color: #6b7280; text-align: center; margin: 18px 0; }
    .primary-btn { width: 100%; height: 52px; border-radius: 18px; background: #60a5fa; border: none; color: white; font-size: 16px; font-weight: 600; cursor: pointer; }
    
    /* Dynamic styling for result */
    .result-text { font-weight: bold; }
    .result-text.match { color: #16a34a; } /* Green for match */
    .result-text.no-match { color: #dc2626; } /* Red for no match */
  </style>
</head>
<body>

<div class="app">
  <div class="result-container">

    <!-- Header -->
    <div class="top-bar">
      <a href="javascript:history.back()" class="icon-btn">←</a>
      <h2>{{ organ or 'Prediction' }} Results</h2>
      <div class="icon-btn">⤴</div>
    </div>

    <!-- Success Probability -->
    {% if probability is defined %}
    <div class="card probability">
      <h3>Success Probability</h3>
      <div class="score">{{ (probability * 100) | round(0) | int }}%</div>
      <p>This score represents the likelihood of a successful transplant.</p>
    </div>
    {% endif %}

    <!-- Prediction Outcome -->
    <div class="section-title">Prediction Outcome</div>
    <div class="card">
        <h4 style="font-size:14px; text-align: center;">
            The model predicts this is: 
            <span class="result-text {{ 'match' if result == 'MATCH' else 'no-match' }}">
                {{ result or 'N/A' }}
            </span>
        </h4>
    </div>
    
    <!-- Disclaimer -->
    <p class="disclaimer">
      Disclaimer: These results are for informational purposes only and are not a substitute for professional medical advice. The static breakdown below is for demonstration only.
    </p>

    <!-- Static Breakdown (as in original file) -->
    <div class="section-title">Detailed Breakdown (Static Demo)</div>
    <div class="card">
      <h4 style="font-size:14px;margin-bottom:12px;">Key Factors</h4>
      <div class="factor"><span>Tissue Compatibility</span><div class="bar"><div class="bar-fill" style="width: 95%;"></div></div></div>
      <div class="factor"><span>Blood Type Match</span><div class="bar"><div class="bar-fill" style="width: 90%;"></div></div></div>
      <div class="factor"><span>Patient Age Factor</span><div class="bar"><div class="bar-fill" style="width: 78%;"></div></div></div>
    </div>
    <div class="dual-cards">
      <div class="mini-card"><h4>Risk Level</h4><div class="big">Low</div><p>Minimal complication risk identified.</p></div>
      <div class="mini-card"><h4>Recovery</h4><div class="big">12–16</div><p>Weeks projected</p></div>
    </div>

    <!-- CTA -->
    <button class="primary-btn" onclick="alert('This would lead to a consultation booking page.')">Consult a Specialist</button>

  </div>
</div>

</body>
</html>
