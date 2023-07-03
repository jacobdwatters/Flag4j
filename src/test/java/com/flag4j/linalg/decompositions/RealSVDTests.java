package com.flag4j.linalg.decompositions;

import com.flag4j.Matrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class RealSVDTests {

    final SingularValueDecomposition<Matrix> svd = new RealSVD();

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
                {-0.04026943724457477, -0.005584143138527626, 0.999173253129513},
                {0.36991542638438507, -0.9290146207773394, 0.009716568571038737}
        };
        expU = new Matrix(expUEntries);
        expSEntries = new double[][]{
                {103.99805783777956, 0.0, 0.0, 0.0},
                {0.0, 100.10197909864775, 0.0, 0.0},
                {0.0, 0.0, 24.807778140450107, 0.0}
        };
        expS = new Matrix(expSEntries);
        expVEntries = new double[][]{
                {0.3263266903138458, -0.9445323500485615, 0.03373745699863595, 0.0},
                {0.9065375992803092, 0.31521260516179134, 0.16027287490914896, 0.0},
                {-0.15371637544368746, -0.01694486795547118, 0.9862632805045105, 0.0},
                {-0.21925270397502555, -0.09062298112967755, 0.021427344494895625, 0.0}
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
                {-0.08547311955880307, 0.8938804065462778, 0.0, 0.0},
                {-1.0712993718750234E-4, -0.2398537404164441, 0.0, 0.0},
                {-0.9823523886528838, -0.013647949167425498, 0.0, 0.0},
                {0.16636742128204293, 0.3784993203222771, 0.0, 0.0}
        };
        expU = new Matrix(expUEntries);
        expSEntries = new double[][]{
                {1174.1202987429022, 0.0},
                {0.0, 38.544566009654424},
                {0.0, 0.0},
                {0.0, 0.0}
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
                {-0.08410500753631646, 0.16399263880284157, 0.2356569943861632, 0.29550452425305634, 0.34053422335445604, 0.36848811454978847, 0.3779644730092278, -0.36848811454978847, -0.34053422335445804, -0.295504524253052, 0.23565699438616358, -0.16399263880284062, -0.08410500753631951},
                {-0.16399263880283638, 0.2955045242530534, 0.3684881145497891, 0.36848811454979186, 0.29550452425304935, 0.16399263880284015, -4.903591898216093E-16, 0.16399263880284046, 0.29550452425305185, 0.36848811454978947, -0.36848811454978936, 0.2955045242530514, 0.1639926388028411},
                {-0.23565699438615986, 0.368488114549792, 0.340534223354458, 0.1639926388028398, -0.08410500753632072, -0.2955045242530516, -0.377964473009227, 0.29550452425305185, 0.08410500753631983, -0.163992638802841, 0.3405342233544578, -0.3684881145497886, -0.23565699438616408},
                {-0.2955045242530486, 0.36848811454979236, 0.16399263880284076, -0.16399263880284376, -0.3684881145497876, -0.29550452425305096, -8.078336605244448E-17, -0.29550452425305124, -0.36848811454978975, -0.16399263880284096, -0.16399263880284093, 0.36848811454978864, 0.29550452425305207},
                {-0.34053422335445577, 0.29550452425305423, -0.08410500753631867, -0.36848811454979, -0.23565699438616047, 0.1639926388028414, 0.37796447300922736, -0.16399263880284143, 0.23565699438616358, 0.3684881145497892, -0.08410500753631922, -0.2955045242530508, -0.3405342233544581},
                {-0.3684881145497884, 0.16399263880284293, -0.2955045242530508, -0.2955045242530496, 0.1639926388028436, 0.36848811454978975, 4.70422413741508E-16, 0.3684881145497893, 0.16399263880284126, -0.295504524253052, 0.2955045242530513, 0.16399263880283968, 0.3684881145497896},
                {-0.37796447300922825, 1.4307510117118142E-15, -0.37796447300922664, 3.471347026657358E-15, 0.377964473009227, 4.403715321828415E-16, -0.37796447300922736, 5.014243497223995E-16, -0.3779644730092273, 7.17476343107559E-16, -0.377964473009227, 1.1096156613569503E-15, -0.37796447300922736},
                {-0.36848811454979147, -0.1639926388028388, -0.29550452425305046, 0.29550452425305285, 0.1639926388028385, -0.3684881145497894, -2.74441661034831E-16, -0.3684881145497892, 0.1639926388028406, 0.29550452425305107, 0.29550452425305207, -0.16399263880284165, 0.3684881145497887},
                {-0.34053422335446104, -0.29550452425304924, -0.08410500753631725, 0.368488114549787, -0.23565699438616663, -0.16399263880284107, 0.3779644730092272, 0.16399263880284062, 0.23565699438616386, -0.36848811454978925, -0.08410500753631955, 0.29550452425305235, -0.34053422335445743},
                {-0.2955045242530543, -0.36848811454978614, 0.16399263880284298, 0.1639926388028378, -0.3684881145497901, 0.29550452425305174, 5.447168242019454E-16, 0.2955045242530522, -0.36848811454978925, 0.1639926388028412, -0.16399263880284096, -0.3684881145497899, 0.29550452425305146},
                {-0.23565699438616575, -0.36848811454978647, 0.3405342233544595, -0.16399263880284154, -0.08410500753631815, 0.2955045242530522, -0.37796447300922686, -0.29550452425305196, 0.08410500753631912, 0.16399263880284018, 0.34053422335445793, 0.3684881145497898, -0.23565699438616322},
                {-0.16399263880284223, -0.29550452425304924, 0.36848811454978986, -0.3684881145497868, 0.2955045242530547, -0.16399263880284076, -2.2804943653384614E-16, -0.16399263880284115, 0.2955045242530512, -0.36848811454978886, -0.36848811454978964, -0.2955045242530522, 0.16399263880284068},
                {-0.08410500753632014, -0.16399263880283946, 0.23565699438616397, -0.29550452425304946, 0.34053422335446054, -0.3684881145497894, 0.37796447300922686, 0.36848811454978897, -0.3405342233544572, 0.2955045242530519, 0.23565699438616353, 0.16399263880284104, -0.08410500753631917}
        };
        expU = new Matrix(expUEntries);
        expSEntries = new double[][]{
                {3.9498558243636537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 3.801937735804853, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 3.5636629649360536, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 3.2469796037174747, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 2.867767478235105, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 2.4450418679126233, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.9999999999999991, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.55495813208737, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1322325217648816, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7530203962825343, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.43633703506394017, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.19806226419516149, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.050144175636352976}
        };
        expS = new Matrix(expSEntries);
        expVEntries = new double[][]{
                {-0.08410500753631663, 0.16399263880284187, 0.23565699438616344, 0.29550452425305546, 0.3405342233544561, 0.36848811454978875, 0.37796447300922703, -0.36848811454978825, -0.3405342233544583, -0.29550452425305207, 0.23565699438616364, -0.1639926388028408, -0.08410500753632015},
                {-0.1639926388028369, 0.2955045242530527, 0.36848811454978886, 0.36848811454979175, 0.2955045242530495, 0.16399263880284054, -4.726664541090788E-16, 0.16399263880284062, 0.295504524253052, 0.3684881145497895, -0.3684881145497891, 0.2955045242530517, 0.16399263880284218},
                {-0.23565699438616036, 0.36848811454979197, 0.3405342233544576, 0.16399263880283949, -0.08410500753632068, -0.29550452425305096, -0.3779644730092272, 0.2955045242530517, 0.08410500753631982, -0.16399263880284115, 0.3405342233544578, -0.36848811454978875, -0.23565699438616525},
                {-0.295504524253049, 0.3684881145497931, 0.16399263880284087, -0.16399263880284476, -0.3684881145497876, -0.2955045242530513, 2.998860456710814E-17, -0.2955045242530512, -0.3684881145497895, -0.16399263880284073, -0.16399263880284104, 0.3684881145497885, 0.2955045242530528},
                {-0.3405342233544563, 0.29550452425305473, -0.0841050075363192, -0.36848811454979014, -0.23565699438616072, 0.1639926388028416, 0.3779644730092273, -0.16399263880284168, 0.2356569943861635, 0.3684881145497894, -0.08410500753631925, -0.29550452425305074, -0.34053422335445843},
                {-0.3684881145497878, 0.16399263880284318, -0.2955045242530511, -0.2955045242530494, 0.1639926388028428, 0.3684881145497899, 4.168895686074974E-16, 0.36848811454978897, 0.16399263880284143, -0.29550452425305185, 0.2955045242530512, 0.16399263880283996, 0.36848811454978936},
                {-0.3779644730092273, 2.4286314138696467E-15, -0.37796447300922625, 3.55486521955989E-15, 0.3779644730092267, -1.9476761425170473E-16, -0.3779644730092275, 7.724238920107815E-16, -0.37796447300922714, 2.604101243663178E-16, -0.3779644730092268, 1.044105625643099E-15, -0.37796447300922725},
                {-0.3684881145497917, -0.16399263880283838, -0.2955045242530498, 0.2955045242530529, 0.16399263880283776, -0.36848811454978936, -2.851615515293359E-17, -0.36848811454978975, 0.16399263880284068, 0.2955045242530513, 0.29550452425305174, -0.1639926388028418, 0.36848811454978897},
                {-0.3405342233544612, -0.2955045242530493, -0.08410500753631699, 0.3684881145497867, -0.23565699438616677, -0.16399263880284126, 0.3779644730092276, 0.16399263880284054, 0.23565699438616375, -0.3684881145497886, -0.08410500753632023, 0.29550452425305246, -0.3405342233544572},
                {-0.29550452425305435, -0.36848811454978647, 0.16399263880284354, 0.16399263880283776, -0.3684881145497901, 0.2955045242530514, 2.998475620465682E-16, 0.29550452425305196, -0.368488114549789, 0.16399263880284135, -0.1639926388028406, -0.3684881145497898, 0.29550452425305107},
                {-0.23565699438616552, -0.3684881145497861, 0.34053422335446, -0.16399263880284173, -0.08410500753631782, 0.29550452425305196, -0.3779644730092268, -0.295504524253052, 0.08410500753631941, 0.16399263880284012, 0.3405342233544581, 0.3684881145497896, -0.23565699438616217},
                {-0.16399263880284215, -0.2955045242530494, 0.36848811454979, -0.36848811454978736, 0.2955045242530551, -0.1639926388028405, -3.0669609634205523E-16, -0.1639926388028413, 0.29550452425305124, -0.36848811454978897, -0.36848811454978997, -0.295504524253052, 0.1639926388028399},
                {-0.08410500753632012, -0.16399263880283954, 0.23565699438616391, -0.29550452425304985, 0.3405342233544609, -0.3684881145497896, 0.3779644730092271, 0.36848811454978936, -0.34053422335445765, 0.29550452425305157, 0.23565699438616422, 0.16399263880284093, -0.08410500753631889}
        };
        expV = new Matrix(expVEntries);

        svd.decompose(A);
        assertEquals(expU, svd.getU());
        assertEquals(expS, svd.getS());
        assertEquals(expV, svd.getV());
    }
}
