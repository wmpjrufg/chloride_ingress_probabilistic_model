// =======================
// Quadrado externo + 3 furos quadrados (50 pontos por furo)
// =======================

// -----------------------
// Parâmetros
// -----------------------
L = 100;                 // tamanho do quadrado externo (0..L)
lc_out = 4;              // malha "grossa" fora
lc_hole = 1.0;           // malha mais fina nas bordas dos furos
n = 50;                  // pontos por furo (50)

xc[] = {30, 70, 50};     // centros X dos 3 furos
yc[] = {30, 30, 70};     // centros Y dos 3 furos
a[]  = { 8, 10,  9};     // semi-lado de cada quadrado (metade do lado)

// -----------------------
// Quadrado externo
// -----------------------
p1 = newp; Point(p1) = {0, 0, 0, lc_out};
p2 = newp; Point(p2) = {L, 0, 0, lc_out};
p3 = newp; Point(p3) = {L, L, 0, lc_out};
p4 = newp; Point(p4) = {0, L, 0, lc_out};

l1 = newl; Line(l1) = {p1, p2};
l2 = newl; Line(l2) = {p2, p3};
l3 = newl; Line(l3) = {p3, p4};
l4 = newl; Line(l4) = {p4, p1};

loopOut = newll; Line Loop(loopOut) = {l1, l2, l3, l4};

// -----------------------
// Função paramétrica para “quadrado” com n pontos
// Mapeia t ∈ [0,4) para as 4 arestas do quadrado (sentido horário)
// Começa em (xc-a, yc+a) e percorre topo → direita → base → esquerda
// -----------------------
holesLL[] = {}; // guardaremos aqui os Line Loops dos furos

For h In {0:#xc[]-1}
  xc_h = xc[h];
  yc_h = yc[h];
  a_h  = a[h];

  pts[] = {};
  // cria n pontos ao redor do quadrado
  For k In {0:n-1}
    t  = 4.0 * k / n;
    If (t < 1)                       // topo: esquerda -> direita
      px = xc_h - a_h + (2*a_h) * t;
      py = yc_h + a_h;
    ElseIf (t < 2)                   // direita: topo -> base
      px = xc_h + a_h;
      py = yc_h + a_h - (2*a_h) * (t - 1);
    ElseIf (t < 3)                   // base: direita -> esquerda
      px = xc_h + a_h - (2*a_h) * (t - 2);
      py = yc_h - a_h;
    Else                             // esquerda: base -> topo
      px = xc_h - a_h;
      py = yc_h - a_h + (2*a_h) * (t - 3);
    EndIf
    pid = newp; Point(pid) = {px, py, 0, lc_hole};
    pts[k] = pid;
  EndFor

  // cria as linhas conectando os n pontos e fecha no final
  lines[] = {};
  For k In {0:n-1}
    pA = pts[k];
    pB = pts[(k+1)%n];
    lid = newl; Line(lid) = {pA, pB};
    lines[k] = lid;
  EndFor

  // loop do furo
  ll = newll; Line Loop(ll) = {lines[]};
  holesLL[h] = ll;
EndFor

// -----------------------
// Superfície com furos
// -----------------------
sur = news;
Plane Surface(sur) = {loopOut, holesLL[]};

// -----------------------
// (Opcional) Controle de malha
// -----------------------
Mesh.Algorithm = 6; // Frontal-Delaunay
// Ajuste global (opcional, pode comentar se preferir os lc dos pontos)
Mesh.CharacteristicLengthMin = Min(lc_hole, lc_out);
Mesh.CharacteristicLengthMax = Max(lc_hole, lc_out);

// Para ver a malha diretamente ao abrir (opcional)
// Mesh 2;
