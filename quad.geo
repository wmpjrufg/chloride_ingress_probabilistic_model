//------------------------------------------------------------
// Domínio 2500x2500 com 50 furos quadrados (semi-lado=40)
// Distribuídos aleatoriamente, sem checar colisão
// Malha quad-dominant fora dos furos
//------------------------------------------------------------

L       = 2500;   // lado do quadrado externo
lc_out  = 300;    // malha grossa fora
lc_hole = 60;     // malha mais fina nos furos
n       = 12;     // pontos por furo (contorno poligonal)
a       = 40;     // semi-lado de cada quadrado interno
Nholes  = 50;     // número de furos

// (opcional) semente p/ reproducibilidade
// General.RandomSeed = 123;

// -----------------------
// Quadrado externo
// -----------------------
p1 = newp; Point(p1) = {0, 0, 0, lc_out};
p2 = newp; Point(p2) = {L, 0, 0, lc_out};
p3 = newp; Point(p3) = {L, L, 0, lc_out};
p4 = newp; Point(p4) = {0, L, 0, lc_out};

l1 = newl; Line(l1) = {p1,p2};
l2 = newl; Line(l2) = {p2,p3};
l3 = newl; Line(l3) = {p3,p4};
l4 = newl; Line(l4) = {p4,p1};

loopOut = newll; Line Loop(loopOut) = {l1,l2,l3,l4};

// -----------------------
// Criação aleatória dos 50 furos
// -----------------------
holesLL[] = {};

For h In {0:Nholes-1}
  xc = a + (L - 2*a) * Rand();
  yc = a + (L - 2*a) * Rand();

  pts[] = {};
  For k In {0:n-1}
    t = 4.0 * k / n;
    If (t < 1)            // topo
      px = xc - a + 2*a*t; py = yc + a;
    ElseIf (t < 2)        // direita
      px = xc + a; py = yc + a - 2*a*(t-1);
    ElseIf (t < 3)        // base
      px = xc + a - 2*a*(t-2); py = yc - a;
    Else                  // esquerda
      px = xc - a; py = yc - a + 2*a*(t-3);
    EndIf
    pid = newp; Point(pid) = {px, py, 0, lc_hole};
    pts[k] = pid;
  EndFor

  lines[] = {};
  For k In {0:n-1}
    pA = pts[k];
    pB = pts[(k+1)%n];
    lid = newl; Line(lid) = {pA,pB};
    lines[k] = lid;
  EndFor

  ll = newll; Line Loop(ll) = {lines[]};
  holesLL[] += {ll};
EndFor

// -----------------------
// Superfície com furos
// -----------------------
sur = news;
Plane Surface(sur) = {loopOut, holesLL[]};

// -----------------------
// Malha quad-dominant
// -----------------------
Recombine Surface{sur};
Mesh.RecombineAll = 1;
Mesh.RecombinationAlgorithm = 1;
Mesh.Algorithm = 6;

Mesh.CharacteristicLengthFromPoints = 1;
Mesh.CharacteristicLengthFromCurvature = 0;
Mesh.CharacteristicLengthExtendFromBoundary = 0;
Mesh.CharacteristicLengthMin = Min(lc_hole, lc_out);
Mesh.CharacteristicLengthMax = Max(lc_hole, lc_out);

Physical Surface("dominio") = {sur};
Physical Curve("borda_externa") = {l1,l2,l3,l4};

Mesh 2;
