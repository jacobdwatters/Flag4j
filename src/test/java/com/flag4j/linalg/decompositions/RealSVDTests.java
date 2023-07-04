package com.flag4j.linalg.decompositions;

import com.flag4j.Matrix;
import com.flag4j.Vector;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class RealSVDTests {

    final SVD<Matrix> svd = new RealSVD(true);

    double[][] aEntries, expSEntries, expUEntries, expVEntries;

    Matrix A, expS, expU, expV;

    @Test
    void svdTestCase() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        };
        A = new Matrix(aEntries);

        expUEntries = new double[][]{
                {-0.21483723836839622, 0.8872306883463706, -0.4082482904638626},
                {-0.520587389464737, 0.24964395298829764, 0.8164965809277261},
                {-0.8263375405610778, -0.3879427823697744, -0.40824829046386324}
        };
        expU = new Matrix(expUEntries);
        expSEntries = new double[][]{
                {16.848103352614167, 0.0, 0.0},
                {0.0, 1.0683695145547083, 0.0},
                {0.0, 0.0, 1.1023900701150984E-16}
        };
        expS = new Matrix(expSEntries);
        expVEntries = new double[][]{
                {-0.47967117787777175, -0.7766909903215589, -0.40824829046386213},
                {-0.5723677939720622, -0.07568647010455855, 0.8164965809277265},
                {-0.6650644100663531, 0.6253180501124429, -0.4082482904638633}
        };
        expV = new Matrix(expVEntries);

        svd.decompose(A);
        assertEquals(expU, svd.getU());
        assertEquals(expS, svd.getS());
        assertEquals(expV, svd.getV());

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[][]{
                {3.45, -99.34, 14.5, 24.5},
                {-0.0024, 0, 25.1, 1.5},
                {100.4, 5.6, -4.1, -0.002}
        };
        A = new Matrix(aEntries);
        expUEntries = new double[][]{
                {-0.9281923021372587, -0.3700008807115799, -0.039476556660765744},
                {-0.04026943724457477, -0.005584143138527625, 0.999173253129513},
                {0.36991542638438507, -0.9290146207773394, 0.009716568571038737}
        };
        expU = new Matrix(expUEntries);
        expSEntries = new double[][]{
                {103.99805783777956, 0.0, 0.0},
                {0.0, 100.1019790986477, 0.0},
                {0.0, 0.0, 24.807778140450107}
        };
        expS = new Matrix(expSEntries);
        expVEntries = new double[][]{
                {0.3263266903138458, -0.9445323500485617, 0.03373745699863595},
                {0.9065375992803092, 0.3152126051617914, 0.16027287490914896},
                {-0.15371637544368746, -0.016944867955471194, 0.9862632805045105},
                {-0.21925270397502555, -0.09062298112967758, 0.021427344494895625}
        };
        expV = new Matrix(expVEntries);

        svd.decompose(A);
        assertEquals(expU, svd.getU());
        assertEquals(expS, svd.getS());
        assertEquals(expV, svd.getV());

        // -------------------- Sub-case 3 --------------------
        aEntries = new double[][]{
                {34.5, 100.34},
                {-9.245, 0.13},
                {0, 1153.4},
                {14.5, -195.342}
        };
        A = new Matrix(aEntries);
        expUEntries = new double[][]{
                {-0.08547311955880307, 0.8938804065462778},
                {-1.0712993718750234E-4, -0.2398537404164441},
                {-0.9823523886528838, -0.013647949167425498},
                {0.16636742128204293, 0.3784993203222771}
        };
        expU = new Matrix(expUEntries);
        expSEntries = new double[][]{
                {1174.1202987429022, 0.0},
                {0.0, 38.544566009654424}
        };
        expS = new Matrix(expSEntries);
        expVEntries = new double[][]{
                {-4.5609006206009825E-4, 0.9999998959909222},
                {-0.9999998959909222, -4.560900620602084E-4}
        };
        expV = new Matrix(expVEntries);

        svd.decompose(A);
        assertEquals(expU, svd.getU());
        assertEquals(expS, svd.getS());
        assertEquals(expV, svd.getV());

        // -------------------- Sub-case 4 --------------------
        // This Toeplitz matrix is known to be difficult to compute eigenvalues of. As such, it is a good test matrix.
        aEntries = new double[][]{
                {2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2},
        };
        A = new Matrix(aEntries);
        expUEntries = new double[][]{
                {-0.08410500753631646, 0.16399263880284157, -0.2356569943861632, 0.2955045242530564, -0.340534223354456, 0.3684881145497885, 0.3779644730092278, 0.3684881145497886, -0.34053422335445804, 0.29550452425305207, -0.2356569943861636, 0.16399263880284065, -0.08410500753631951},
                {-0.16399263880283638, 0.2955045242530534, -0.36848811454978914, 0.36848811454979186, -0.2955045242530493, 0.16399263880284015, -4.903591898216088E-16, -0.16399263880284048, 0.29550452425305185, -0.36848811454978947, 0.36848811454978936, -0.2955045242530514, 0.1639926388028411},
                {-0.23565699438615986, 0.368488114549792, -0.340534223354458, 0.16399263880283974, 0.08410500753632077, -0.2955045242530516, -0.37796447300922714, -0.29550452425305196, 0.08410500753631985, 0.16399263880284098, -0.3405342233544579, 0.36848811454978847, -0.23565699438616408},
                {-0.2955045242530486, 0.36848811454979236, -0.16399263880284076, -0.16399263880284376, 0.3684881145497875, -0.29550452425305096, -8.078336605244356E-17, 0.29550452425305135, -0.36848811454978964, 0.16399263880284096, 0.16399263880284093, -0.3684881145497887, 0.29550452425305207},
                {-0.34053422335445577, 0.2955045242530543, 0.08410500753631868, -0.3684881145497901, 0.23565699438616036, 0.1639926388028414, 0.3779644730092274, 0.16399263880284146, 0.23565699438616364, -0.36848811454978925, 0.08410500753631922, 0.29550452425305074, -0.3405342233544581},
                {-0.3684881145497884, 0.16399263880284293, 0.29550452425305085, -0.2955045242530496, -0.16399263880284365, 0.36848811454978975, 4.704224137415076E-16, -0.3684881145497894, 0.1639926388028413, 0.295504524253052, -0.2955045242530513, -0.1639926388028397, 0.3684881145497896},
                {-0.37796447300922825, 1.4307510117117302E-15, 0.37796447300922675, 3.554661207201604E-15, -0.3779644730092269, 4.403715321828403E-16, -0.3779644730092274, -5.014243497224006E-16, -0.37796447300922736, -7.174763431075588E-16, 0.377964473009227, -1.1096156613569532E-15, -0.37796447300922736},
                {-0.36848811454979147, -0.1639926388028388, 0.29550452425305046, 0.29550452425305285, -0.16399263880283838, -0.3684881145497893, -2.7444166103483087E-16, 0.3684881145497893, 0.1639926388028406, -0.2955045242530511, -0.29550452425305207, 0.16399263880284165, 0.3684881145497887},
                {-0.34053422335446104, -0.29550452425304924, 0.08410500753631725, 0.3684881145497869, 0.23565699438616663, -0.16399263880284107, 0.37796447300922725, -0.16399263880284065, 0.23565699438616391, 0.36848811454978925, 0.08410500753631957, -0.2955045242530524, -0.34053422335445743},
                {-0.2955045242530543, -0.36848811454978625, -0.16399263880284298, 0.16399263880283774, 0.36848811454979, 0.29550452425305174, 5.447168242019443E-16, -0.2955045242530523, -0.36848811454978925, -0.1639926388028412, 0.16399263880284096, 0.3684881145497899, 0.29550452425305146},
                {-0.23565699438616575, -0.36848811454978647, -0.34053422335445954, -0.1639926388028416, 0.08410500753631811, 0.2955045242530522, -0.37796447300922686, 0.295504524253052, 0.08410500753631912, -0.16399263880284018, -0.34053422335445793, -0.3684881145497898, -0.23565699438616322},
                {-0.16399263880284223, -0.29550452425304924, -0.36848811454978986, -0.3684881145497868, -0.2955045242530546, -0.16399263880284073, -2.2804943653384717E-16, 0.16399263880284118, 0.2955045242530512, 0.36848811454978886, 0.36848811454978964, 0.29550452425305224, 0.16399263880284068},
                {-0.08410500753632014, -0.16399263880283946, -0.235656994386164, -0.29550452425304946, -0.3405342233544605, -0.3684881145497894, 0.37796447300922686, -0.3684881145497891, -0.3405342233544572, -0.2955045242530519, -0.2356569943861635, -0.16399263880284107, -0.08410500753631917}
        };
        expU = new Matrix(expUEntries);
        expSEntries = new double[][]{
                {3.9498558243636537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 3.801937735804854, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 3.563662964936053, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 3.246979603717475, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 2.8677674782351046, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 2.4450418679126233, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.9999999999999996, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5549581320873698, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1322325217648816, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7530203962825341, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4363370350639403, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.19806226419516149, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.050144175636352976}
        };
        expS = new Matrix(expSEntries);
        expVEntries = new double[][]{
                {-0.08410500753631663, 0.16399263880284187, -0.23565699438616344, 0.29550452425305546, -0.34053422335445604, 0.3684881145497887, 0.37796447300922703, 0.3684881145497883, -0.34053422335445827, 0.2955045242530521, -0.23565699438616364, 0.16399263880284082, -0.08410500753632015},
                {-0.1639926388028369, 0.2955045242530527, -0.3684881145497889, 0.3684881145497918, -0.29550452425304946, 0.1639926388028405, -4.726664541090785E-16, -0.16399263880284065, 0.295504524253052, -0.3684881145497895, 0.3684881145497891, -0.2955045242530516, 0.16399263880284218},
                {-0.23565699438616036, 0.368488114549792, -0.3405342233544576, 0.16399263880283946, 0.08410500753632066, -0.2955045242530509, -0.3779644730092272, -0.29550452425305174, 0.0841050075363198, 0.16399263880284118, -0.3405342233544578, 0.36848811454978875, -0.23565699438616525},
                {-0.295504524253049, 0.3684881145497931, -0.1639926388028409, -0.16399263880284476, 0.36848811454978747, -0.29550452425305124, 2.9988604567107056E-17, 0.29550452425305124, -0.3684881145497895, 0.16399263880284076, 0.16399263880284107, -0.3684881145497885, 0.2955045242530528},
                {-0.3405342233544563, 0.29550452425305473, 0.08410500753631922, -0.36848811454979025, 0.23565699438616067, 0.16399263880284157, 0.3779644730092273, 0.1639926388028417, 0.23565699438616344, -0.36848811454978936, 0.08410500753631926, 0.29550452425305074, -0.34053422335445843},
                {-0.3684881145497878, 0.16399263880284318, 0.2955045242530511, -0.2955045242530494, -0.1639926388028428, 0.3684881145497899, 4.1688956860749773E-16, -0.36848811454978897, 0.1639926388028414, 0.29550452425305196, -0.29550452425305124, -0.16399263880283996, 0.36848811454978936},
                {-0.3779644730092273, 2.4286314138697E-15, 0.3779644730092263, 3.56505735525595E-15, -0.3779644730092267, -1.9476761425170695E-16, -0.3779644730092276, -7.724238920107818E-16, -0.3779644730092271, -2.6041012436631824E-16, 0.37796447300922675, -1.044105625643097E-15, -0.37796447300922725},
                {-0.3684881145497917, -0.16399263880283838, 0.2955045242530498, 0.295504524253053, -0.16399263880283776, -0.36848811454978936, -2.851615515293316E-17, 0.3684881145497899, 0.16399263880284068, -0.2955045242530513, -0.2955045242530518, 0.16399263880284176, 0.36848811454978897},
                {-0.3405342233544612, -0.2955045242530493, 0.084105007536317, 0.3684881145497867, 0.23565699438616677, -0.16399263880284123, 0.37796447300922764, -0.16399263880284054, 0.23565699438616372, 0.36848811454978864, 0.08410500753632025, -0.29550452425305246, -0.3405342233544572},
                {-0.29550452425305435, -0.3684881145497864, -0.16399263880284357, 0.16399263880283776, 0.3684881145497901, 0.29550452425305135, 2.998475620465679E-16, -0.29550452425305196, -0.36848811454978897, -0.16399263880284137, 0.16399263880284062, 0.3684881145497898, 0.29550452425305107},
                {-0.23565699438616552, -0.36848811454978614, -0.34053422335446, -0.1639926388028418, 0.0841050075363178, 0.2955045242530519, -0.3779644730092268, 0.29550452425305207, 0.08410500753631941, -0.16399263880284015, -0.3405342233544581, -0.3684881145497896, -0.23565699438616217},
                {-0.16399263880284215, -0.2955045242530494, -0.3684881145497901, -0.3684881145497874, -0.29550452425305507, -0.1639926388028405, -3.0669609634205425E-16, 0.16399263880284132, 0.29550452425305124, 0.3684881145497891, 0.36848811454978997, 0.295504524253052, 0.1639926388028399},
                {-0.08410500753632012, -0.1639926388028395, -0.23565699438616394, -0.29550452425304985, -0.3405342233544608, -0.36848811454978947, 0.3779644730092271, -0.36848811454978936, -0.34053422335445754, -0.2955045242530516, -0.23565699438616428, -0.1639926388028409, -0.08410500753631889}
        };
        expV = new Matrix(expVEntries);

        svd.decompose(A);
        assertEquals(expU, svd.getU());
        assertEquals(expS, svd.getS());
        assertEquals(expV, svd.getV());
    }
}
