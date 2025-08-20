// .geo by json contours
SetFactory("Built-in");
Geometry.Tolerance = 1e-10;

// Add aggregate 03
Point(1) = { 1, 1, 0, 0.3 };
Point(2) = { 3, 2, 0, 0.3 };
Point(3) = { 2, 3, 0, 0.3 };
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 1};
Line Loop(1) = {1, 2, 3};
Plane Surface(1) = {1};

// Add aggregate 04
Point(4) = { 4, 1, 0, 0.3 };
Point(5) = { 7.5, 4, 0, 0.3 };
Point(6) = { 6, 7, 0, 0.3 };
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 4};
Line Loop(2) = {4, 5, 6};
Plane Surface(2) = {2};

// Add aggregate 05
Point(7) = { 8, 6, 0, 0.3 };
Point(8) = { 9, 6, 0, 0.3 };
Point(9) = { 9, 5, 0, 0.3 };
Line(7) = {7, 8};
Line(8) = {8, 9};
Line(9) = {9, 7};
Line Loop(3) = {7, 8, 9};
Plane Surface(3) = {3};

// Add aggregate 06
Point(10) = { 2, 9, 0, 0.3 };
Point(11) = { 3, 8, 0, 0.3 };
Point(12) = { 1, 7, 0, 0.3 };
Line(10) = {10, 11};
Line(11) = {11, 12};
Line(12) = {12, 10};
Line Loop(4) = {10, 11, 12};
Plane Surface(4) = {4};
Physical Surface("aggregate") = {1, 2, 3, 4};

// Add rebar
Point(13) = { 7.0, 8.5, 0, 0.3 };
Point(14) = { 8.0, 8.5, 0, 0.3 };
Point(15) = { 8.0, 9.5, 0, 0.3 };
Point(16) = { 7.0, 9.5, 0, 0.3 };
Line(13) = {13, 14};
Line(14) = {14, 15};
Line(15) = {15, 16};
Line(16) = {16, 13};
Line Loop(5) = {13, 14, 15, 16};
Plane Surface(5) = {5};
Physical Surface("rebar") = {5};

// Add mortar
Point(17) = { 0, 0, 0, 0.3 };
Point(18) = { 10, 0, 0, 0.3 };
Point(19) = { 10, 10, 0, 0.3 };
Point(20) = { 0, 10, 0, 0.3 };
Line(17) = {17, 18};
Line(18) = {18, 19};
Line(19) = {19, 20};
Line(20) = {20, 17};
Line Loop(6) = {17, 18, 19, 20};
Plane Surface(6) = {1, 2, 3, 4, 5, 6};
Physical Surface("mortar") = {6};

Mesh 2;
Coherence Mesh;
Coherence;
Save "output_contour.geo";