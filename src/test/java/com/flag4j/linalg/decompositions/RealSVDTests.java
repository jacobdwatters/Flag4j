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
                {0.21483723836839627, -0.8872306883463714, -8.060878704448943E-8},
                {0.520587389464737, -0.2496439529882993, -2.388408505021909E-8},
                {0.8263375405610779, 0.3879427823697723, 3.582612757532863E-8}
        };
        expU = new Matrix(expUEntries);
        expSEntries = new double[][]{
                {16.848103352614206, 0.0, 0.0},
                {0.0, 1.0683695145547074, 0.0},
                {0.0, 0.0, 7.437407946192002E-8}
        };
        expS = new Matrix(expSEntries);
        expVEntries = new double[][]{
                {0.4796711778777717, 0.7766909903215564, 0.4082482904638677},
                {0.5723677939720622, 0.07568647010456324, -0.8164965809277254},
                {0.6650644100663529, -0.6253180501124449, 0.4082482904638591}
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
                {-0.9281923021372513, -0.3700008807115994, -0.03947655666076524},
                {-0.040269437244574664, -0.005584143138528235, 0.9991732531295133},
                {0.36991542638440306, -0.9290146207773315, 0.009716568571038722}
        };
        expU = new Matrix(expUEntries);
        expSEntries = new double[][]{
                {103.99805783777148, 0.0, 0.0, 0.0},
                {0.0, 100.10197909864621, 0.0, 0.0},
                {0.0, 0.0, 24.80777814045011, 0.0}
        };
        expS = new Matrix(expSEntries);
        expVEntries = new double[][]{
                {0.3263266903138674, -0.9445323500485558, 0.03373745699863596, -0.015209039937366706},
                {0.9065375992803042, 0.31521260516180916, 0.16027287490914857, 0.23052808984279555},
                {-0.15371637544368502, -0.016944867955473925, 0.9862632805045107, -0.05804213037879165},
                {-0.21925270397502505, -0.090622981129682, 0.0214273444948958, 0.9712139805412126}
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
                {-0.8938804065460952, 0.08547311955880338, 0.0, 0.0},
                {0.2398537404163953, 1.0712993718765095E-4, 0.0, 0.0},
                {0.01364794916742265, 0.9823523886528839, 0.0, 0.0},
                {-0.3784993203222, -0.16636742128204293, 0.0, 0.0}
        };
        expU = new Matrix(expUEntries);
        expSEntries = new double[][]{
                {38.54456600966224, 0.0},
                {0.0, 1174.1202987429047},
                {0.0, 0.0},
                {0.0, 0.0}
        };
        expS = new Matrix(expSEntries);
        expVEntries = new double[][]{
                {-0.999999895990922, 4.5609006206020206E-4},
                {4.5609006206020206E-4, 0.9999998959909223}
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
                {0.08410500753632054, 0.1639926388028389, 0.23565699438616594, 0.2955045242530514, -0.3405342233544581, 0.36848811454978864, -0.37796447300922825, 0.36848811454978875, 0.3405342233544575, 0.2955045242530513, 0.2356569943861634, 0.16399263880284315, 0.08410500753632251},
                {0.16399263880284304, 0.2955045242530481, 0.36848811454979236, 0.36848811454978814, -0.295504524253052, 0.16399263880284062, -1.1102230246251568E-16, -0.16399263880284115, -0.29550452425305196, -0.36848811454978825, -0.36848811454979, -0.295504524253058, -0.16399263880284815},
                {0.2356569943861669, 0.3684881145497845, 0.3405342233544604, 0.16399263880283974, 0.0841050075363199, -0.29550452425305107, 0.3779644730092281, -0.295504524253051, -0.08410500753631857, 0.16399263880284085, 0.34053422335445793, 0.36848811454979236, 0.23565699438617108},
                {0.2955045242530557, 0.3684881145497848, 0.16399263880284198, -0.16399263880284135, 0.3684881145497903, -0.2955045242530511, 1.1102230246251568E-16, 0.2955045242530523, 0.3684881145497888, 0.16399263880284018, -0.1639926388028396, -0.3684881145497965, -0.2955045242530686},
                {0.3405342233544617, 0.2955045242530487, -0.08410500753631926, -0.36848811454978864, 0.2356569943861642, 0.16399263880284065, -0.3779644730092278, 0.16399263880283982, -0.23565699438616466, -0.3684881145497887, -0.08410500753632019, 0.2955045242530548, 0.3405342233544782},
                {0.36848811454979147, 0.16399263880283926, -0.2955045242530514, -0.2955045242530505, -0.163992638802842, 0.3684881145497889, 1.1102230246251568E-16, -0.3684881145497896, -0.16399263880284015, 0.2955045242530517, 0.2955045242530524, -0.1639926388028417, -0.3684881145498099},
                {0.37796447300922714, -1.0220526626708326E-15, -0.3779644730092263, 2.0002603917256725E-15, -0.37796447300922853, 2.0433202704534353E-16, 0.37796447300922814, 7.853879161674125E-16, 0.3779644730092279, -2.2115397473415876E-16, -0.3779644730092281, -1.4013560699417203E-16, 0.3779644730092847},
                {0.3684881145497868, -0.1639926388028423, -0.2955045242530513, 0.295504524253054, -0.16399263880284137, -0.36848811454978925, 8.881784197001254E-16, 0.36848811454978897, -0.1639926388028412, -0.2955045242530514, 0.2955045242530518, 0.16399263880284173, -0.3684881145498719},
                {0.3405342233544543, -0.29550452425305435, -0.08410500753632047, 0.36848811454979086, 0.2356569943861638, -0.16399263880284148, -0.3779644730092267, -0.16399263880284165, -0.2356569943861634, 0.36848811454978964, -0.08410500753631751, -0.2955045242530541, 0.3405342233545568},
                {0.2955045242530481, -0.3684881145497933, 0.16399263880283863, 0.16399263880284173, 0.3684881145497894, 0.295504524253052, -2.2204460492503136E-16, -0.2955045242530519, 0.3684881145497894, -0.16399263880284085, -0.16399263880284265, 0.3684881145497901, -0.2955045242531483},
                {0.2356569943861607, -0.3684881145497945, 0.34053422335445516, -0.1639926388028411, 0.08410500753631969, 0.2955045242530527, 0.37796447300922653, 0.2955045242530513, -0.08410500753631985, -0.1639926388028416, 0.34053422335445865, -0.3684881145497876, 0.23565699438622753},
                {0.16399263880283882, -0.29550452425305684, 0.3684881145497865, -0.36848811454978986, -0.29550452425305174, -0.16399263880284087, -1.1102230246251568E-16, 0.16399263880284065, -0.29550452425305196, 0.3684881145497898, -0.36848811454978786, 0.29550452425304474, -0.16399263880287776},
                {0.0841050075363184, -0.16399263880284393, 0.23565699438616208, -0.2955045242530524, -0.3405342233544581, -0.3684881145497899, -0.37796447300922675, -0.36848811454978847, 0.34053422335445865, -0.2955045242530516, 0.235656994386163, -0.16399263880283585, 0.08410500753633635}
        };
        expU = new Matrix(expUEntries);
        expSEntries = new double[][]{
                {3.9498558243636452, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 3.8019377358048345, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 3.5636629649360656, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 3.2469796037174654, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 2.8677674782351112, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 2.4450418679126282, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.9999999999999998, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5549581320873707, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1322325217648839, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7530203962825328, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4363370350639402, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.19806226419516074, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.050144175636344775}
        };
        expS = new Matrix(expSEntries);
        expVEntries = new double[][]{
                {0.08410500753632062, 0.16399263880283885, 0.2356569943861662, 0.2955045242530514, -0.3405342233544574, 0.3684881145497887, -0.37796447300922814, 0.36848811454978897, 0.34053422335445804, 0.2955045242530511, 0.2356569943861634, 0.16399263880284357, 0.08410500753631599},
                {0.16399263880284273, 0.29550452425304774, 0.3684881145497936, 0.36848811454978814, -0.29550452425305174, 0.16399263880284032, -7.00271783648284E-17, -0.1639926388028413, -0.29550452425305246, -0.36848811454978825, -0.36848811454978864, -0.29550452425305684, -0.16399263880283452},
                {0.23565699438616705, 0.3684881145497841, 0.3405342233544604, 0.16399263880283946, 0.08410500753631957, -0.29550452425305096, 0.3779644730092281, -0.2955045242530511, -0.08410500753631843, 0.16399263880284046, 0.34053422335445693, 0.3684881145497957, 0.23565699438615537},
                {0.29550452425305546, 0.3684881145497843, 0.16399263880284232, -0.1639926388028411, 0.36848811454978975, -0.295504524253051, 1.1382428745203288E-17, 0.29550452425305224, 0.36848811454978875, 0.16399263880284082, -0.16399263880283946, -0.3684881145497969, -0.29550452425304297},
                {0.34053422335446204, 0.2955045242530485, -0.08410500753631994, -0.36848811454978847, 0.23565699438616394, 0.16399263880284035, -0.37796447300922786, 0.1639926388028399, -0.23565699438616428, -0.3684881145497891, -0.08410500753632072, 0.2955045242530597, 0.34053422335445016},
                {0.36848811454979163, 0.1639926388028394, -0.29550452425305207, -0.2955045242530507, -0.16399263880284193, 0.3684881145497888, 1.665944400659758E-16, -0.36848811454978936, -0.16399263880284046, 0.29550452425305207, 0.29550452425305257, -0.16399263880284862, -0.36848811454978414},
                {0.37796447300922675, -6.959244929256197E-16, -0.37796447300922664, 1.825925419231809E-15, -0.37796447300922775, 3.9286743346370057E-16, 0.37796447300922775, 9.513874063560301E-16, 0.3779644730092278, -7.572067972638763E-16, -0.37796447300922736, 7.555005340949272E-15, 0.3779644730092269},
                {0.3684881145497864, -0.1639926388028419, -0.2955045242530515, 0.2955045242530535, -0.1639926388028408, -0.3684881145497891, 3.9928455507209915E-16, 0.3684881145497887, -0.1639926388028417, -0.29550452425305074, 0.2955045242530509, 0.1639926388028335, -0.36848811454979363},
                {0.340534223354454, -0.2955045242530537, -0.08410500753632065, 0.3684881145497906, 0.23565699438616342, -0.16399263880284154, -0.37796447300922675, -0.16399263880284146, -0.23565699438616303, 0.36848811454978814, -0.08410500753631761, -0.2955045242530445, 0.34053422335446604},
                {0.29550452425304774, -0.3684881145497929, 0.1639926388028389, 0.16399263880284148, 0.36848811454978864, 0.29550452425305174, -1.8487002293593457E-16, -0.2955045242530513, 0.3684881145497889, -0.1639926388028396, -0.16399263880284282, 0.36848811454978186, -0.2955045242530613},
                {0.23565699438616056, -0.36848811454979435, 0.34053422335445593, -0.1639926388028411, 0.08410500753631964, 0.29550452425305285, 0.3779644730092267, 0.2955045242530513, -0.0841050075363193, -0.16399263880284243, 0.3405342233544592, -0.36848811454978203, 0.23565699438617216},
                {0.1639926388028388, -0.2955045242530568, 0.3684881145497873, -0.36848811454978975, -0.29550452425305135, -0.16399263880284096, -1.719138069747328E-16, 0.16399263880284048, -0.2955045242530523, 0.3684881145497904, -0.36848811454978947, 0.2955045242530455, -0.16399263880284692},
                {0.08410500753631839, -0.16399263880284387, 0.23565699438616242, -0.2955045242530522, -0.3405342233544575, -0.3684881145497899, -0.3779644730092266, -0.3684881145497883, 0.3405342233544586, -0.29550452425305235, 0.23565699438616372, -0.16399263880283718, 0.08410500753632254}
        };
        expV = new Matrix(expVEntries);

        svd.decompose(A);
        assertEquals(expU, svd.getU());
        assertEquals(expS, svd.getS());
        assertEquals(expV, svd.getV());
    }
}
