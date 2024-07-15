package org.flag4j.sparse_matrix;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class CooMatrixMultTransposeTests {

    @Test
    void realSparseMultTransposeTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        Shape bShape;
        int[] bRowIndices;
        int[] bColIndices;
        double[] bEntries;
        CooMatrix b;

        double[][] expEntries;
        Matrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.78156, 0.09594, 0.7923};
        aRowIndices = new int[]{0, 1, 2};
        aColIndices = new int[]{1, 0, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(6, 5);
        bEntries = new double[]{0.46839, 0.47218, 0.85592, 0.41846, 0.03665, 0.40249, 0.39273, 0.71011, 0.50029, 0.19742};
        bRowIndices = new int[]{0, 1, 2, 2, 3, 3, 3, 3, 4, 4};
        bColIndices = new int[]{3, 4, 1, 2, 0, 2, 3, 4, 3, 4};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new double[][]{
                {0.0, 0.0, 0.6689528352, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.003516201, 0.0, 0.0},
                {0.0, 0.0, 0.678145416, 0.0, 0.0, 0.0}};
        exp = new Matrix(expEntries);

        assertTrue(exp.allClose(a.multTranspose(b)));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new double[]{0.99702, 0.04209, 0.43944, 0.33732, 0.37757, 0.05866, 0.89726, 0.68715, 0.32244, 0.352, 0.47304, 0.41871, 0.49412, 0.88239, 0.77977};
        aRowIndices = new int[]{0, 2, 2, 3, 4, 4, 5, 5, 5, 5, 6, 6, 7, 7, 10};
        aColIndices = new int[]{21, 4, 18, 10, 12, 15, 12, 18, 19, 20, 9, 15, 3, 10, 9};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(11, 23);
        bEntries = new double[]{0.43453, 0.96681, 0.53232, 0.42268, 0.55315, 0.32311, 0.36187, 0.41291, 0.05022, 0.31466, 0.16399, 0.38665, 0.04099, 0.55214, 0.09856, 0.16123, 0.07109, 0.18844, 0.68079, 0.44251, 0.48795, 0.90615, 0.27059, 0.91353, 0.16297, 0.83766, 0.98706, 0.71687, 0.78636, 0.15918, 0.69246, 0.3795, 0.39076, 0.00326, 0.21866, 0.08403, 0.53308, 0.79918, 0.44156, 0.58684, 0.62729, 0.00474, 0.94979, 0.65794, 0.63977, 0.95383, 0.87742, 0.40367, 0.61562, 0.16512, 0.81519, 0.5108, 0.45016, 0.28453, 0.65645, 0.45985, 0.95643, 0.23393, 0.38601, 0.58675, 0.32708, 0.04307, 0.35009, 0.34327, 0.11326, 0.35421, 0.86255, 0.37221, 0.07691, 0.03179, 0.66814, 0.38309, 0.9688, 0.3194, 0.99001, 0.57288, 0.08684, 0.84867, 0.19332, 0.61244, 0.16635, 0.38489, 0.61532, 0.96326, 0.48203, 0.18518, 0.44577, 0.78922, 0.85329};
        bRowIndices = new int[]{0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10};
        bColIndices = new int[]{0, 5, 6, 8, 14, 19, 21, 5, 7, 8, 17, 18, 19, 20, 0, 1, 4, 5, 10, 14, 15, 20, 21, 9, 10, 16, 21, 0, 9, 10, 19, 20, 22, 0, 3, 4, 6, 7, 13, 15, 20, 21, 1, 4, 5, 6, 9, 11, 13, 14, 21, 22, 1, 3, 5, 7, 14, 15, 18, 21, 0, 1, 6, 9, 10, 11, 15, 16, 21, 0, 2, 3, 7, 12, 13, 14, 15, 18, 19, 20, 21, 22, 1, 8, 13, 14, 15, 20, 21};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new double[][]{
                {0.36079162740000004, 0.0, 0.26978364180000003, 0.9841185612000001, 0.0, 0.0047258748, 0.8127607338, 0.585001485, 0.0766808082, 0.165854277, 0.8507471958},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.169909476, 0.0029921781000000003, 0.0, 0.0, 0.0035368227, 0.0276926946, 0.1696282344, 0.0, 0.3729395448, 0.0},
                {0.0, 0.0, 0.2296440828, 0.054973040400000006, 0.0536945976, 0.0, 0.0, 0.0, 0.0382048632, 0.0, 0.0},
                {0.0, 0.0, 0.028623146999999998, 0.0, 0.0, 0.0344240344, 0.0, 0.0137223338, 0.050597183, 0.1256898924, 0.026148868199999998},
                {0.1041835884, 0.47325664310000004, 0.3189648, 0.0, 0.35686080239999995, 0.22080608, 0.0, 0.26524677150000003, 0.0, 1.1476614153000002, 0.27780544},
                {0.0, 0.0, 0.20430954450000002, 0.4321362312, 0.3719797344, 0.24571577640000003, 0.4150547568, 0.09794883030000001, 0.5235387513, 0.0363607764, 0.18664835670000002},
                {0.0, 0.0, 0.6007222881000001, 0.14380309830000002, 0.1404588402, 0.1080442792, 0.0, 0.1405919636, 0.0999394914, 0.1892924308, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.7123432880999999, 0.6131799372, 0.0, 0.6841857934, 0.0, 0.2676716479, 0.0, 0.0}
        };
        exp = new Matrix(expEntries);

        assertTrue(exp.allClose(a.multTranspose(b)));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.6657, 0.789, 0.34576, 0.67106};
        aRowIndices = new int[]{1, 2, 3, 4};
        aColIndices = new int[]{0, 0, 0, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(1, 3);
        bEntries = new double[]{0.40693};
        bRowIndices = new int[]{0};
        bColIndices = new int[]{1};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new double[][]{
                {0.0},
                {0.0},
                {0.0},
                {0.0},
                {0.0}};
        exp = new Matrix(expEntries);

        assertTrue(exp.allClose(a.multTranspose(b)));

        // ---------------------  Sub-case 4 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.63359, 0.98973, 0.65753, 0.27274};
        aRowIndices = new int[]{1, 2, 3, 4};
        aColIndices = new int[]{2, 1, 0, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(2, 3);
        bEntries = new double[]{0.25708, 0.02006};
        bRowIndices = new int[]{0, 0};
        bColIndices = new int[]{0, 2};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new double[][]{
                {0.0, 0.0},
                {0.012709815400000001, 0.0},
                {0.0, 0.0},
                {0.16903781239999996, 0.0},
                {0.07011599919999999, 0.0}};
        exp = new Matrix(expEntries);

        assertTrue(exp.allClose(a.multTranspose(b)));

        // ---------------------  Sub-case 5 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.43181, 0.80241, 0.7987, 0.961};
        aRowIndices = new int[]{1, 2, 4, 4};
        aColIndices = new int[]{0, 0, 0, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 2);
        bEntries = new double[]{0.79177, 0.7031, 0.58915, 0.31236};
        bRowIndices = new int[]{0, 1, 3, 4};
        bColIndices = new int[]{1, 0, 1, 1};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        CooMatrix final0a = a;
        CooMatrix final0b = b;
        assertThrows(Exception.class, ()->final0a.multTranspose(final0b));

        // ---------------------  Sub-case 6 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.28394, 0.65788, 0.63941, 0.47642};
        aRowIndices = new int[]{0, 2, 4, 4};
        aColIndices = new int[]{1, 2, 0, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 5);
        bEntries = new double[]{0.23905, 0.29491, 0.73915, 0.58077, 0.92534, 0.0505, 0.64781, 0.49145, 0.58197};
        bRowIndices = new int[]{0, 0, 0, 1, 2, 2, 2, 3, 4};
        bColIndices = new int[]{0, 2, 4, 2, 1, 2, 3, 2, 4};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        CooMatrix final1a = a;
        CooMatrix final1b = b;
        assertThrows(Exception.class, ()->final1a.multTranspose(final1b));
    }


    @Test
    void complexSparseMultTransposeTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        Shape bShape;
        int[] bRowIndices;
        int[] bColIndices;
        CNumber[] bEntries;
        CooCMatrix b;

        CNumber[][] expEntries;
        CMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.22693, 0.22917, 0.13093};
        aRowIndices = new int[]{1, 1, 2};
        aColIndices = new int[]{2, 3, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(6, 5);
        bEntries = new CNumber[]{new CNumber("0.3866+0.63381i"), new CNumber("0.83231+0.02353i"), new CNumber("0.61932+0.1677i"), new CNumber("0.76325+0.17661i"), new CNumber("0.76246+0.08687i"), new CNumber("0.83761+0.53999i"), new CNumber("0.49695+0.08075i"), new CNumber("0.01463+0.99901i"), new CNumber("0.76917+0.70453i"), new CNumber("0.88303+0.62023i")};
        bRowIndices = new int[]{0, 0, 0, 1, 1, 2, 3, 3, 5, 5};
        bColIndices = new int[]{1, 3, 4, 2, 4, 4, 2, 4, 0, 4};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new CNumber[][]{
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.19074048270000002+0.0053923701i"), new CNumber("0.17320432249999998+0.0400781073i"), new CNumber("0.0"), new CNumber("0.1127728635+0.0183245975i"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.050617538+0.0829847433i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")}};
        exp = new CMatrix(expEntries);

        assertTrue(exp.allClose(a.multTranspose(b)));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new double[]{0.18616, 0.61944, 0.01393, 0.12385, 0.52623, 0.38444, 0.39158, 0.28796, 0.67486, 0.32967, 0.53034, 0.39248, 0.93527, 0.71114, 0.91245};
        aRowIndices = new int[]{0, 0, 0, 0, 2, 3, 4, 7, 8, 9, 9, 10, 10, 10, 10};
        aColIndices = new int[]{8, 12, 13, 16, 8, 6, 21, 8, 18, 5, 14, 6, 10, 14, 15};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(11, 23);
        bEntries = new CNumber[]{new CNumber("0.11326+0.21529i"), new CNumber("0.16093+0.37556i"), new CNumber("0.60587+0.68942i"), new CNumber("0.08898+0.40518i"), new CNumber("0.73738+0.54495i"), new CNumber("0.05558+0.47171i"), new CNumber("0.39592+0.07556i"), new CNumber("0.66512+0.7311i"), new CNumber("0.75171+0.00983i"), new CNumber("0.14495+0.65308i"), new CNumber("0.85428+0.98408i"), new CNumber("0.7162+0.4342i"), new CNumber("0.99183+0.27161i"), new CNumber("0.01706+0.36868i"), new CNumber("0.64447+0.42982i"), new CNumber("0.11732+0.14036i"), new CNumber("0.22487+0.25962i"), new CNumber("0.20117+0.59076i"), new CNumber("0.21979+0.78763i"), new CNumber("0.74715+0.30403i"), new CNumber("0.3254+0.58496i"), new CNumber("0.2241+0.78881i"), new CNumber("0.9414+0.43149i"), new CNumber("0.43824+0.74881i"), new CNumber("0.5222+0.9847i"), new CNumber("0.30911+0.8953i"), new CNumber("0.83735+0.37187i"), new CNumber("0.39008+0.99501i"), new CNumber("0.39649+0.47272i"), new CNumber("0.67448+0.45517i"), new CNumber("0.15781+0.22518i"), new CNumber("0.65499+0.24462i"), new CNumber("0.01747+0.83417i"), new CNumber("0.11765+0.70957i"), new CNumber("0.641+0.57914i"), new CNumber("0.73416+0.46681i"), new CNumber("0.20806+0.59924i"), new CNumber("0.80886+0.89294i"), new CNumber("0.36172+0.47479i"), new CNumber("0.35518+0.11242i"), new CNumber("0.10965+0.1513i"), new CNumber("0.71181+0.92229i"), new CNumber("0.92651+0.28278i"), new CNumber("0.22448+0.31465i"), new CNumber("0.37724+0.11213i"), new CNumber("0.07311+0.4702i"), new CNumber("0.97149+0.28604i"), new CNumber("0.03766+0.87296i"), new CNumber("0.78616+0.3937i"), new CNumber("0.22906+0.08291i"), new CNumber("0.83272+0.49961i"), new CNumber("0.62723+0.27856i"), new CNumber("0.17572+0.00659i"), new CNumber("0.70165+0.70007i"), new CNumber("0.27449+0.3353i"), new CNumber("0.39763+0.70191i"), new CNumber("0.71936+0.35985i"), new CNumber("0.64845+0.47984i"), new CNumber("0.63461+0.63074i"), new CNumber("0.60546+0.69088i"), new CNumber("0.12219+0.32181i"), new CNumber("0.19233+0.30391i"), new CNumber("0.38694+0.82648i"), new CNumber("0.3372+0.01379i"), new CNumber("0.89179+0.56752i"), new CNumber("0.59521+0.09615i"), new CNumber("0.63307+0.86818i"), new CNumber("0.19163+0.97831i"), new CNumber("0.03125+0.93526i"), new CNumber("0.55349+0.89289i"), new CNumber("0.85297+0.54658i"), new CNumber("0.32775+0.44111i"), new CNumber("0.94849+0.41837i"), new CNumber("0.32257+0.49672i"), new CNumber("0.60249+0.71227i"), new CNumber("0.52567+0.34761i"), new CNumber("0.51186+0.36647i"), new CNumber("0.29525+0.89198i"), new CNumber("0.16237+0.58546i"), new CNumber("0.24183+0.63519i"), new CNumber("0.09985+0.73333i"), new CNumber("0.28215+0.16499i"), new CNumber("0.93032+0.12596i"), new CNumber("0.87588+0.67933i"), new CNumber("0.754+0.32481i"), new CNumber("0.61457+0.2482i"), new CNumber("0.6621+0.77779i"), new CNumber("0.70686+0.42471i"), new CNumber("0.65432+0.40042i")};
        bRowIndices = new int[]{0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10};
        bColIndices = new int[]{0, 1, 2, 6, 9, 13, 20, 1, 3, 6, 7, 12, 17, 1, 5, 6, 14, 22, 1, 3, 5, 8, 10, 12, 13, 14, 15, 18, 0, 6, 7, 8, 9, 12, 14, 18, 20, 21, 0, 3, 4, 9, 12, 19, 0, 1, 2, 3, 4, 7, 8, 10, 11, 12, 17, 18, 19, 22, 0, 4, 6, 9, 18, 20, 3, 5, 6, 7, 10, 11, 13, 17, 7, 8, 12, 13, 14, 17, 19, 3, 6, 10, 11, 12, 15, 18, 19, 20, 22};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new CNumber[][]{
                {new CNumber(7.742293999999999E-4, 0.0065709203), new CNumber(0.44364292799999994, 0.268960848), new CNumber(0.0, 0.0), new CNumber(0.3204560876, 0.624404607), new CNumber(0.1948100544, 0.4850745), new CNumber(0.5739173543999999, 0.1751652432), new CNumber(0.5896492312, 0.5266587584), new CNumber(0.0, 0.0), new CNumber(0.0118818721, 0.007613859399999999), new CNumber(0.44057861989999997, 0.5385201312999999), new CNumber(0.5425551072, 0.42080417519999996)},
                {new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0)},
                {new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.11792814299999999, 0.4150954863), new CNumber(0.34467538769999995, 0.1287263826), new CNumber(0.0, 0.0), new CNumber(0.4382022456, 0.2629097703), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.1697460111, 0.26138896559999997), new CNumber(0.0, 0.0)},
                {new CNumber(0.0342074712, 0.1557673992), new CNumber(0.055724578, 0.2510700752), new CNumber(0.0451025008, 0.05395999840000001), new CNumber(0.0, 0.0), new CNumber(0.2592970912, 0.1749855548), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0469747236, 0.1237166364), new CNumber(0.24337743080000002, 0.33376311919999996), new CNumber(0.0, 0.0), new CNumber(0.038386334, 0.28192138520000004)},
                {new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.3167333988, 0.34965744519999997), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0)},
                {new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0)},
                {new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0)},
                {new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.064531836, 0.22714572759999999), new CNumber(0.1886109204, 0.0704407752), new CNumber(0.0, 0.0), new CNumber(0.2397900512, 0.14386769559999998), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0928872572, 0.1430354912), new CNumber(0.0, 0.0)},
                {new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.2632493888, 0.6714924486), new CNumber(0.49545521760000005, 0.3150313966), new CNumber(0.0, 0.0), new CNumber(0.2683445818, 0.47369098260000003), new CNumber(0.2611303284, 0.5577582928), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.4147487102, 0.16750025200000002)},
                {new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.3317199807, 0.27938563020000007), new CNumber(0.27120801540000006, 0.6676571652000001), new CNumber(0.33994794, 0.3071411076), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.19622288070000002, 0.0316977705), new CNumber(0.2714598324, 0.19435369980000003), new CNumber(0.0, 0.0)},
                {new CNumber(0.0349228704, 0.15902504639999998), new CNumber(0.056889975999999995, 0.2563208384), new CNumber(0.20595980539999997, 0.23971465960000002), new CNumber(1.8643236709000002, 1.3795560758), new CNumber(0.7205606504, 0.5904947412), new CNumber(0.0, 0.0), new CNumber(0.5866294021, 0.2605288112), new CNumber(0.047957131199999996, 0.1263039888), new CNumber(0.27769450110000005, 1.2154639066000001), new CNumber(0.36400412039999996, 0.2606114758), new CNumber(0.9910628585, 0.7385004401999999)}
        };
        exp = new CMatrix(expEntries);

        assertTrue(exp.allClose(a.multTranspose(b)));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.58314, 0.11483, 0.07423, 0.21489};
        aRowIndices = new int[]{1, 3, 4, 4};
        aColIndices = new int[]{0, 0, 1, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(1, 3);
        bEntries = new CNumber[]{new CNumber("0.04283+0.03142i")};
        bRowIndices = new int[]{0};
        bColIndices = new int[]{2};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new CNumber[][]{
                {new CNumber("0.0")},
                {new CNumber("0.0")},
                {new CNumber("0.0")},
                {new CNumber("0.0")},
                {new CNumber("0.0092037387+0.006751843799999999i")}};
        exp = new CMatrix(expEntries);

        assertTrue(exp.allClose(a.multTranspose(b)));

        // ---------------------  Sub-case 4 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.98502, 0.84417, 0.30885, 0.79799};
        aRowIndices = new int[]{1, 2, 4, 4};
        aColIndices = new int[]{2, 0, 0, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(2, 3);
        bEntries = new CNumber[]{new CNumber("0.60355+0.44762i"), new CNumber("0.44255+0.25246i")};
        bRowIndices = new int[]{0, 1};
        bColIndices = new int[]{2, 2};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new CNumber[][]{
                {new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.594508821+0.4409146524i"), new CNumber("0.435920601+0.24867814920000003i")},
                {new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.48162686450000003+0.3571962838i"), new CNumber("0.3531504745+0.20146055540000002i")}};
        exp = new CMatrix(expEntries);

        assertTrue(exp.allClose(a.multTranspose(b)));

        // ---------------------  Sub-case 5 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.52864, 0.31344, 0.02112, 0.5396};
        aRowIndices = new int[]{0, 2, 2, 3};
        aColIndices = new int[]{2, 0, 1, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 2);
        bEntries = new CNumber[]{new CNumber("0.53805+0.69145i"), new CNumber("0.13068+0.18815i"), new CNumber("0.04164+0.25769i"), new CNumber("0.58659+0.5605i")};
        bRowIndices = new int[]{1, 2, 3, 3};
        bColIndices = new int[]{0, 0, 0, 1};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        CooMatrix final0a = a;
        CooCMatrix final0b = b;
        assertThrows(Exception.class, ()->final0a.multTranspose(final0b));

        // ---------------------  Sub-case 6 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.2073, 0.3688, 0.55774, 0.1757};
        aRowIndices = new int[]{0, 1, 2, 2};
        aColIndices = new int[]{0, 1, 0, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 5);
        bEntries = new CNumber[]{new CNumber("0.32298+0.72064i"), new CNumber("0.6846+0.41292i"), new CNumber("0.04506+0.47792i"), new CNumber("0.38245+0.26152i"), new CNumber("0.96951+0.92176i"), new CNumber("0.21641+0.22405i"), new CNumber("0.1344+0.85491i"), new CNumber("0.86766+0.17831i"), new CNumber("0.58265+0.82329i")};
        bRowIndices = new int[]{0, 0, 1, 2, 2, 3, 3, 4, 4};
        bColIndices = new int[]{1, 3, 0, 0, 1, 2, 3, 0, 2};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        CooMatrix final1a = a;
        CooCMatrix final1b = b;
        assertThrows(Exception.class, ()->final1a.multTranspose(final1b));
    }


    @Test
    void realDenseMultTransposeTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        double[][] bEntries;
        Matrix b;

        double[][] expEntries;
        Matrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.30789, 0.95793, 0.64079};
        aRowIndices = new int[]{0, 1, 2};
        aColIndices = new int[]{4, 3, 3};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.01617, 0.73997, 0.88414, 0.57132, 0.48007},
                {0.30189, 0.06066, 0.3781, 0.29104, 0.94529},
                {0.35441, 0.01424, 0.21778, 0.05035, 0.41897},
                {0.38596, 0.21208, 0.29123, 0.25912, 0.20182},
                {0.45528, 0.28327, 0.16233, 0.29626, 0.40903},
                {0.39479, 0.90299, 0.08433, 0.04361, 0.23269}};
        b = new Matrix(bEntries);

        expEntries = new double[][]{
                {0.14780875229999998, 0.2910453381, 0.1289966733, 0.0621383598, 0.1259362467, 0.0716429241},
                {0.5472845676, 0.2787959472, 0.0482317755, 0.2482188216, 0.2837963418, 0.0417753273},
                {0.3660961428, 0.1864955216, 0.0322637765, 0.16604150480000002, 0.1898404454, 0.0279448519}};
        exp = new Matrix(expEntries);

        assertTrue(exp.allClose(a.multTranspose(b)));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new double[]{0.64154, 0.76618, 0.16321, 0.05973, 0.62777, 0.27239, 0.06577, 0.4791, 0.96198, 0.0751, 0.92295, 0.81271, 0.46399, 0.36052, 0.93608};
        aRowIndices = new int[]{1, 1, 2, 3, 5, 5, 6, 6, 7, 8, 8, 8, 8, 9, 10};
        aColIndices = new int[]{9, 12, 6, 6, 3, 15, 1, 5, 21, 8, 16, 17, 22, 10, 18};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.15428, 0.66933, 0.28201, 0.40696, 0.66793, 0.86561, 0.45816, 0.74211, 0.9461, 0.63664, 0.32587, 0.34212, 0.41204, 0.80125, 0.07786, 0.07574, 0.42959, 0.66955, 0.37806, 0.98863, 0.80339, 0.08339, 0.58958},
                {0.47474, 0.22358, 0.66662, 0.87621, 0.72689, 0.17063, 0.39965, 0.75274, 0.56788, 0.68219, 0.14579, 0.47687, 0.50735, 0.03227, 0.07268, 0.57986, 0.54573, 0.50932, 0.27522, 0.19191, 0.06242, 0.2889, 0.73735},
                {0.9988, 0.75748, 0.86216, 0.99478, 0.72179, 0.87825, 0.60131, 0.54705, 0.56856, 0.51981, 0.89158, 0.95243, 0.20479, 0.33398, 0.08606, 0.99342, 0.48848, 0.54303, 0.3224, 0.30847, 0.08614, 0.21828, 0.60721},
                {0.41153, 0.15051, 0.37696, 0.26517, 5e-05, 0.99407, 0.26075, 0.19682, 0.61871, 0.20122, 0.79593, 0.35204, 0.28998, 0.88952, 0.65212, 0.2833, 0.42343, 0.77311, 0.8955, 0.30184, 0.08993, 0.56858, 0.68753},
                {0.27183, 0.97351, 0.35512, 0.74402, 0.83465, 0.94095, 0.49307, 0.26559, 0.36962, 0.04432, 0.79066, 0.3664, 0.3454, 0.2088, 0.08837, 0.80053, 0.90234, 0.95529, 0.44362, 0.11482, 0.3366, 0.96716, 0.1359},
                {0.46172, 0.47805, 0.82993, 0.59909, 0.89206, 0.68727, 0.53559, 0.34474, 0.26852, 0.45731, 0.01465, 0.10359, 0.93324, 0.85412, 0.63872, 0.84819, 0.60559, 0.51321, 0.3839, 0.62325, 0.31866, 0.1624, 0.58765},
                {0.2124, 0.07172, 0.08281, 0.7888, 0.79336, 0.26825, 0.13231, 0.30336, 0.13616, 0.44695, 0.6558, 0.80721, 0.3986, 0.95959, 0.21438, 0.89867, 0.76714, 0.09352, 0.73249, 0.19584, 0.0097, 0.71093, 0.74631},
                {0.90132, 0.58868, 0.59388, 0.43791, 0.10753, 0.01436, 0.73192, 0.00996, 0.21337, 0.08287, 0.00343, 0.98536, 0.68432, 0.81914, 0.9469, 0.65468, 0.15319, 0.22283, 0.36251, 0.55617, 0.11677, 0.13526, 0.46353},
                {0.61968, 0.54764, 0.04785, 0.33061, 0.15758, 0.203, 0.74066, 0.94795, 0.89656, 0.72529, 0.73731, 0.0558, 0.57096, 0.03074, 0.06991, 0.94417, 0.26127, 0.82386, 0.14188, 0.0189, 0.50469, 0.99271, 0.93772},
                {0.15675, 0.60504, 0.75993, 0.99299, 0.57526, 0.62887, 0.1566, 0.64087, 0.49058, 0.18572, 0.10225, 0.47503, 0.53209, 0.00315, 0.4836, 0.55549, 0.73618, 0.76576, 0.90988, 0.35149, 0.09802, 0.24656, 0.32974},
                {0.2077, 0.2479, 0.53347, 0.61707, 0.0117, 0.15356, 0.90437, 0.04979, 0.50255, 0.40148, 0.85757, 0.17181, 0.30977, 0.82067, 0.40583, 0.42557, 0.11336, 0.53485, 0.71662, 0.82648, 0.6274, 0.39931, 0.9612}};
        b = new Matrix(bEntries);

        expEntries = new double[][]{
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.7241268327999999, 0.8263735956, 0.4903849096, 0.35126755519999997, 0.29307162479999993, 1.0084124805999999, 0.592135651, 0.5774767174, 0.9027606794, 0.526823525, 0.4949050578},
                {0.0747762936, 0.0652268765, 0.0981398051, 0.042557007499999994, 0.0804739547, 0.0874136439, 0.0215943151, 0.1194566632, 0.12088311859999999, 0.025558685999999997, 0.1476022277},
                {0.0273658968, 0.0238710945, 0.0359162463, 0.015574597499999999, 0.0294510711, 0.0319907907, 0.0079028763, 0.0437175816, 0.0442396218, 0.009353717999999999, 0.0540180201},
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                {0.2761080978, 0.7080064171000001, 0.8950907144, 0.24363385790000003, 0.6851298021000001, 0.6071292034000001, 0.7399736973, 0.4532350459000001, 0.46472950600000007, 0.7746792534, 0.5032990462},
                {0.4587355851, 0.0964536896, 0.4705890346, 0.4861579797000001, 0.5148368977, 0.36071240550000006, 0.1332355994, 0.0455973596, 0.13327558280000001, 0.3410850978, 0.089874979},
                {0.0802195122, 0.277916022, 0.20998099439999998, 0.5469625883999999, 0.9303885768, 0.156225552, 0.6839004414, 0.1301174148, 0.9549671658, 0.2371857888, 0.3841282338},
                {1.2852514052, 1.3023817752, 1.2166067512, 1.3845911123, 1.7000031419, 1.2688497651000001, 1.1405424951, 0.5535802515, 1.4131227659, 1.4916367611999999, 1.0230322485},
                {0.1174826524, 0.0525602108, 0.3214324216, 0.28694868360000003, 0.2850487432, 0.005281618, 0.23642901600000002, 0.0012365836, 0.2658150012, 0.03686317, 0.30917113640000005},
                {0.3538944048, 0.2576279376, 0.301792192, 0.83825964, 0.41526380960000003, 0.35936111200000004, 0.6856692392, 0.3393383608, 0.1328110304, 0.8517204704000001, 0.6708136496}
        };
        exp = new Matrix(expEntries);

        assertTrue(exp.allClose(a.multTranspose(b), 0.0005, 0.0005));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.29113, 0.48843, 0.92579, 0.3378};
        aRowIndices = new int[]{1, 2, 3, 4};
        aColIndices = new int[]{0, 1, 1, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.40891, 0.04234, 0.12498}};
        b = new Matrix(bEntries);

        expEntries = new double[][]{
                {0.0},
                {0.1190459683},
                {0.0206801262},
                {0.039197948600000004},
                {0.138129798}};
        exp = new Matrix(expEntries);

        assertTrue(exp.allClose(a.multTranspose(b)));

        // ---------------------  Sub-case 4 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.2342, 0.32307, 0.63419, 0.00208};
        aRowIndices = new int[]{0, 2, 2, 3};
        aColIndices = new int[]{2, 0, 2, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.47401, 0.94996, 0.31452},
                {0.29343, 0.7498, 0.29617}};
        b = new Matrix(bEntries);

        expEntries = new double[][]{
                {0.073660584, 0.069363014},
                {0.0, 0.0},
                {0.3526038495, 0.2826264824},
                {0.0006542016, 0.0006160336},
                {0.0, 0.0}};
        exp = new Matrix(expEntries);

        assertTrue(exp.allClose(a.multTranspose(b)));

        // ---------------------  Sub-case 5 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.26077, 0.0905, 0.70481, 0.75257};
        aRowIndices = new int[]{0, 1, 3, 4};
        aColIndices = new int[]{1, 1, 1, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.06421, 0.75498},
                {0.3979, 0.01385},
                {0.74276, 0.23795},
                {0.89888, 0.20426},
                {0.25745, 0.85686}};
        b = new Matrix(bEntries);

        CooMatrix final0a = a;
        Matrix final0b = b;
        assertThrows(Exception.class, ()->final0a.multTranspose(final0b));

        // ---------------------  Sub-case 6 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.24177, 0.07462, 0.35733, 0.03143};
        aRowIndices = new int[]{0, 3, 3, 4};
        aColIndices = new int[]{1, 1, 2, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.94967, 0.43617, 0.35185, 0.00291, 0.3427},
                {0.01566, 0.58087, 0.48891, 0.95618, 0.28676},
                {0.42614, 0.15461, 0.53321, 0.09456, 0.27216},
                {0.0112, 0.48804, 0.805, 0.81917, 0.20515},
                {0.61793, 0.68949, 0.87053, 0.57533, 0.1838}};
        b = new Matrix(bEntries);

        CooMatrix final1a = a;
        Matrix final1b = b;
        assertThrows(Exception.class, ()->final1a.multTranspose(final1b));
    }


    @Test
    void complexDenseMultTransposeTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        CNumber[][] bEntries;
        CMatrix b;

        CNumber[][] expEntries;
        CMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.08364, 0.13561, 0.0045};
        aRowIndices = new int[]{0, 1, 1};
        aColIndices = new int[]{1, 2, 4};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[][]{
                {new CNumber("0.21341+0.37956i"), new CNumber("0.15156+0.77038i"), new CNumber("0.07565+0.35195i"), new CNumber("0.39581+0.1674i"), new CNumber("0.19116+0.42132i")},
                {new CNumber("0.50815+0.12609i"), new CNumber("0.54215+0.45813i"), new CNumber("0.73503+0.11151i"), new CNumber("0.58787+0.06467i"), new CNumber("0.48954+0.36775i")},
                {new CNumber("0.2723+0.11605i"), new CNumber("0.45158+0.00452i"), new CNumber("0.99315+0.29952i"), new CNumber("0.49616+0.52877i"), new CNumber("0.42593+0.99514i")},
                {new CNumber("0.65213+0.93319i"), new CNumber("0.02188+0.83931i"), new CNumber("0.11212+0.40814i"), new CNumber("0.95391+0.53587i"), new CNumber("0.28471+0.77827i")},
                {new CNumber("0.45394+0.32281i"), new CNumber("0.06687+0.73436i"), new CNumber("0.71263+0.36555i"), new CNumber("0.01161+0.68834i"), new CNumber("0.09807+0.51722i")},
                {new CNumber("0.59029+0.39906i"), new CNumber("0.97974+0.96233i"), new CNumber("0.9539+0.05767i"), new CNumber("0.5223+0.99463i"), new CNumber("0.89229+0.57119i")}};
        b = new CMatrix(bEntries);

        expEntries = new CNumber[][]{
                {new CNumber("0.012676478400000002+0.0644345832i"), new CNumber("0.04534542600000001+0.038317993200000004i"), new CNumber("0.037770151200000004+0.0003780528i"), new CNumber("0.0018300432000000002+0.07019988840000001i"), new CNumber("0.0055930068000000005+0.06142187040000001i"), new CNumber("0.08194545360000001+0.08048928120000001i")},
                {new CNumber("0.0111191165+0.0496238795i"), new CNumber("0.1018803483+0.0167767461i"), new CNumber("0.1365977565+0.0450960372i"), new CNumber("0.0164857882+0.058850080400000004i"), new CNumber("0.0970810693+0.0518997255i"), new CNumber("0.133373684+0.0103909837i")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")}};
        exp = new CMatrix(expEntries);

        assertTrue(exp.allClose(a.multTranspose(b)));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new double[]{0.26027, 0.54714, 0.11642, 0.30499, 0.57117, 0.56418, 0.8048, 0.30036, 0.48682, 0.25393, 0.8547, 0.6226, 0.17137, 0.63381, 0.18879};
        aRowIndices = new int[]{0, 1, 1, 1, 2, 4, 5, 6, 6, 6, 7, 7, 8, 8, 9};
        aColIndices = new int[]{7, 0, 1, 2, 20, 9, 8, 3, 9, 11, 14, 22, 9, 20, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[][]{
                {new CNumber("0.94019+0.45219i"), new CNumber("0.49776+0.69416i"), new CNumber("0.82609+0.18761i"), new CNumber("0.85259+0.39589i"), new CNumber("0.074+0.00208i"), new CNumber("0.65618+0.00136i"), new CNumber("0.10622+0.54856i"), new CNumber("0.63472+0.98929i"), new CNumber("0.19361+0.56297i"), new CNumber("0.05915+0.87057i"), new CNumber("0.96659+0.98726i"), new CNumber("0.4496+0.51708i"), new CNumber("0.64282+0.45275i"), new CNumber("0.21079+0.28304i"), new CNumber("0.81682+0.62875i"), new CNumber("0.46053+0.37044i"), new CNumber("0.74123+0.97676i"), new CNumber("0.07755+0.05617i"), new CNumber("0.32947+0.84318i"), new CNumber("0.86325+0.04581i"), new CNumber("0.52375+0.66078i"), new CNumber("0.53204+0.08248i"), new CNumber("0.87184+0.99764i")},
                {new CNumber("0.61591+0.39612i"), new CNumber("0.19381+0.05779i"), new CNumber("0.46322+0.37511i"), new CNumber("0.02802+0.91155i"), new CNumber("0.64089+0.0564i"), new CNumber("0.2011+0.5583i"), new CNumber("0.8134+0.771i"), new CNumber("0.74442+0.32969i"), new CNumber("0.81989+0.88669i"), new CNumber("0.65485+0.77597i"), new CNumber("0.07159+0.11524i"), new CNumber("0.03344+0.68774i"), new CNumber("0.19293+0.67624i"), new CNumber("0.0757+0.47254i"), new CNumber("0.81422+0.43828i"), new CNumber("0.68579+0.37032i"), new CNumber("0.36672+0.543i"), new CNumber("0.31882+0.13753i"), new CNumber("0.22071+0.98586i"), new CNumber("0.84905+0.94915i"), new CNumber("0.29643+0.69794i"), new CNumber("0.64002+0.30272i"), new CNumber("0.59329+0.51719i")},
                {new CNumber("0.84672+0.10424i"), new CNumber("0.56306+0.47414i"), new CNumber("0.93262+0.90851i"), new CNumber("0.4507+0.22732i"), new CNumber("0.5934+0.69286i"), new CNumber("0.01681+0.05798i"), new CNumber("0.0547+0.42527i"), new CNumber("0.602+0.80734i"), new CNumber("0.13565+0.16262i"), new CNumber("0.73747+0.0951i"), new CNumber("0.38803+0.16829i"), new CNumber("0.43934+0.28052i"), new CNumber("0.98891+0.64878i"), new CNumber("0.85541+0.77735i"), new CNumber("0.35265+0.71945i"), new CNumber("0.25801+0.07817i"), new CNumber("0.77556+0.16093i"), new CNumber("0.34022+0.57877i"), new CNumber("0.683+0.39033i"), new CNumber("0.60561+0.48926i"), new CNumber("0.01774+0.67051i"), new CNumber("0.84528+0.88675i"), new CNumber("0.15081+0.92421i")},
                {new CNumber("0.71069+0.2797i"), new CNumber("0.28809+0.13415i"), new CNumber("0.32039+0.40081i"), new CNumber("0.78127+0.76171i"), new CNumber("0.24932+0.62755i"), new CNumber("0.86109+0.22807i"), new CNumber("0.78633+0.91173i"), new CNumber("0.59833+0.54444i"), new CNumber("0.28051+0.10999i"), new CNumber("0.14013+0.56015i"), new CNumber("0.07784+0.08736i"), new CNumber("0.23373+0.93152i"), new CNumber("0.58041+0.67257i"), new CNumber("0.36596+0.05143i"), new CNumber("0.33505+0.87041i"), new CNumber("0.56974+0.76192i"), new CNumber("0.3238+0.38545i"), new CNumber("0.39994+0.18166i"), new CNumber("0.31311+0.29709i"), new CNumber("0.02113+0.31489i"), new CNumber("0.95738+0.32551i"), new CNumber("0.40571+0.8586i"), new CNumber("0.72852+0.43959i")},
                {new CNumber("0.71863+0.43517i"), new CNumber("0.98329+0.59708i"), new CNumber("0.68379+0.48553i"), new CNumber("0.54646+0.42039i"), new CNumber("0.66137+0.16175i"), new CNumber("0.66587+0.09256i"), new CNumber("0.77511+0.84941i"), new CNumber("0.77343+0.28798i"), new CNumber("0.7101+0.76054i"), new CNumber("0.74333+0.97131i"), new CNumber("0.55017+0.89632i"), new CNumber("0.98601+0.97943i"), new CNumber("0.96871+0.54184i"), new CNumber("0.95253+0.89672i"), new CNumber("0.01169+0.23902i"), new CNumber("0.67707+0.55927i"), new CNumber("0.61958+0.80944i"), new CNumber("0.30641+0.27154i"), new CNumber("0.04728+0.69156i"), new CNumber("0.48659+0.64014i"), new CNumber("0.78799+0.09193i"), new CNumber("0.86575+0.29303i"), new CNumber("0.72715+0.01377i")},
                {new CNumber("0.26375+0.12138i"), new CNumber("0.00185+0.1014i"), new CNumber("0.15151+0.81621i"), new CNumber("0.10021+0.15348i"), new CNumber("0.51508+0.13133i"), new CNumber("0.85464+0.62855i"), new CNumber("0.96804+0.21914i"), new CNumber("0.09911+0.06109i"), new CNumber("0.27341+0.53476i"), new CNumber("0.34964+0.3442i"), new CNumber("0.34311+0.40769i"), new CNumber("0.63933+0.62092i"), new CNumber("0.78819+0.95742i"), new CNumber("0.20305+0.24105i"), new CNumber("0.02837+0.00595i"), new CNumber("0.3866+0.5201i"), new CNumber("0.24207+0.61542i"), new CNumber("0.55429+0.04517i"), new CNumber("0.57386+0.28478i"), new CNumber("0.51034+0.09164i"), new CNumber("0.41975+0.24946i"), new CNumber("0.68939+0.8056i"), new CNumber("0.83776+0.87797i")},
                {new CNumber("0.50805+0.21369i"), new CNumber("0.0941+0.87006i"), new CNumber("0.96457+0.33194i"), new CNumber("0.64996+0.51839i"), new CNumber("0.02122+0.16629i"), new CNumber("0.01713+0.23199i"), new CNumber("0.365+0.54724i"), new CNumber("0.20898+0.09371i"), new CNumber("0.45479+0.10272i"), new CNumber("0.26115+0.99262i"), new CNumber("0.47719+0.47739i"), new CNumber("0.35364+0.74566i"), new CNumber("0.94587+0.50937i"), new CNumber("0.41033+0.50316i"), new CNumber("0.94811+0.24809i"), new CNumber("0.80597+0.42251i"), new CNumber("0.01735+0.48234i"), new CNumber("0.16511+0.94271i"), new CNumber("0.60312+0.62254i"), new CNumber("0.95157+0.61599i"), new CNumber("0.64764+0.10884i"), new CNumber("0.2167+0.77663i"), new CNumber("0.33591+0.77117i")},
                {new CNumber("0.15932+0.49569i"), new CNumber("0.75474+0.20325i"), new CNumber("0.29664+0.80546i"), new CNumber("0.98287+0.11751i"), new CNumber("0.5691+0.04389i"), new CNumber("0.9203+0.25965i"), new CNumber("0.69809+0.90383i"), new CNumber("0.95387+0.76164i"), new CNumber("0.89802+0.17236i"), new CNumber("0.21372+0.88552i"), new CNumber("0.25925+0.49276i"), new CNumber("0.4885+0.69569i"), new CNumber("0.5466+0.28072i"), new CNumber("0.34753+0.98522i"), new CNumber("0.58092+0.83162i"), new CNumber("0.12697+0.98396i"), new CNumber("0.12866+0.50898i"), new CNumber("0.12683+0.88441i"), new CNumber("0.80755+0.60283i"), new CNumber("0.22175+0.89315i"), new CNumber("0.28276+0.23136i"), new CNumber("0.65595+0.92556i"), new CNumber("0.38929+0.0366i")},
                {new CNumber("0.67957+0.41607i"), new CNumber("0.18441+0.66207i"), new CNumber("0.77191+0.61384i"), new CNumber("0.7713+0.50966i"), new CNumber("0.9872+0.34025i"), new CNumber("0.82971+0.65796i"), new CNumber("0.58072+0.53683i"), new CNumber("0.20477+0.92194i"), new CNumber("0.43132+0.23912i"), new CNumber("0.32725+0.19622i"), new CNumber("0.58579+0.98094i"), new CNumber("0.50279+0.36024i"), new CNumber("0.84862+0.02588i"), new CNumber("0.20452+0.05058i"), new CNumber("0.89798+0.72472i"), new CNumber("0.18137+0.4923i"), new CNumber("0.39684+0.72471i"), new CNumber("0.10195+0.27094i"), new CNumber("0.84264+0.68947i"), new CNumber("0.16633+0.03784i"), new CNumber("0.61774+0.23388i"), new CNumber("0.42807+0.38556i"), new CNumber("0.01973+0.41169i")},
                {new CNumber("0.63038+0.74463i"), new CNumber("0.0808+0.8612i"), new CNumber("0.28521+0.20161i"), new CNumber("0.90848+0.71298i"), new CNumber("0.55713+0.48676i"), new CNumber("0.62231+0.79793i"), new CNumber("0.464+0.50198i"), new CNumber("0.99599+0.56513i"), new CNumber("0.61073+0.6863i"), new CNumber("0.16889+0.44194i"), new CNumber("0.80451+0.10574i"), new CNumber("0.81853+0.18333i"), new CNumber("0.79577+0.12547i"), new CNumber("0.62368+0.68077i"), new CNumber("0.82532+0.74359i"), new CNumber("0.10317+0.83118i"), new CNumber("0.89943+0.8364i"), new CNumber("0.50001+0.21765i"), new CNumber("0.71773+0.09085i"), new CNumber("0.59957+0.6978i"), new CNumber("0.68991+0.47537i"), new CNumber("0.12994+0.50262i"), new CNumber("0.4223+0.58027i")},
                {new CNumber("0.49892+0.67518i"), new CNumber("0.49323+0.69i"), new CNumber("0.30396+0.88094i"), new CNumber("0.32717+0.76451i"), new CNumber("0.47272+0.58658i"), new CNumber("0.55634+0.0977i"), new CNumber("0.2044+0.99864i"), new CNumber("0.72865+0.05836i"), new CNumber("0.08311+0.82832i"), new CNumber("0.46214+0.58867i"), new CNumber("0.61373+0.98256i"), new CNumber("0.73817+0.23005i"), new CNumber("0.67456+0.54263i"), new CNumber("0.57678+0.12987i"), new CNumber("0.40524+0.20611i"), new CNumber("0.89585+0.36057i"), new CNumber("0.5116+0.69588i"), new CNumber("0.32654+0.18802i"), new CNumber("0.3453+0.83288i"), new CNumber("0.8868+0.08291i"), new CNumber("0.38143+0.0537i"), new CNumber("0.88167+0.74036i"), new CNumber("0.87657+0.64616i")}};
        b = new CMatrix(bEntries);

        expEntries = new CNumber[][]{
                {new CNumber("0.1651985744+0.2574825083i"), new CNumber("0.1937501934+0.0858084163i"), new CNumber("0.15668254+0.21012638179999998i"), new CNumber("0.1557273491+0.14170139880000002i"), new CNumber("0.2013006261+0.0749525546i"), new CNumber("0.025795359700000003+0.0158998943i"), new CNumber("0.0543912246+0.0243899017i"), new CNumber("0.2482637449+0.1982320428i"), new CNumber("0.0532954879+0.2399533238i"), new CNumber("0.2592263173+0.1470863851i"), new CNumber("0.1896457355+0.015189357200000001i")},
                {new CNumber("0.8243139648999999+0.38544451769999993i"), new CNumber("0.5008298253999999+0.3378658075i"), new CNumber("0.8132655998+0.3893197173i"), new CNumber("0.5201021105+0.2908958429i"), new CNumber("0.7162149520999999+0.4556927621i"), new CNumber("0.19073258689999997+0.3271527291i"), new CNumber("0.5831138033+0.31944911239999996i"), new CNumber("0.2655094092+0.540531437i"), new CNumber("0.6287137729+0.4919417908i"), new CNumber("0.44129904709999995+0.5691667961i"), new CNumber("0.42310568579999996+0.7184256757999999i")},
                {new CNumber("0.2991502875+0.3774177126i"), new CNumber("0.1693119231+0.3986423898i"), new CNumber("0.010132555799999998+0.3829751967i"), new CNumber("0.5468267346+0.1859215467i"), new CNumber("0.4500762482999999+0.05250765809999999i"), new CNumber("0.2397486075+0.14248406819999998i"), new CNumber("0.36991253879999997+0.0621661428i"), new CNumber("0.16150402919999998+0.1321458912i"), new CNumber("0.35283455579999995+0.1335852396i"), new CNumber("0.39405589469999996+0.2715170829i"), new CNumber("0.21786137309999998+0.030671828999999998i")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.033371247+0.4911581826i"), new CNumber("0.369453273+0.43778675460000005i"), new CNumber("0.41606582459999997+0.053653518000000004i"), new CNumber("0.0790585434+0.316025427i"), new CNumber("0.4193719194+0.5479936758i"), new CNumber("0.1972598952+0.19419075600000002i"), new CNumber("0.147335607+0.5600163516i"), new CNumber("0.1205765496+0.4995926736i"), new CNumber("0.184627905+0.1107033996i"), new CNumber("0.09528436020000002+0.2493337092i"), new CNumber("0.2607301452+0.33211584060000005i")},
                {new CNumber("0.155817328+0.45307825599999996i"), new CNumber("0.6598474719999999+0.713608112i"), new CNumber("0.10917111999999998+0.130876576i"), new CNumber("0.22575444799999997+0.088519952i"), new CNumber("0.57148848+0.6120825919999999i"), new CNumber("0.220040368+0.430374848i"), new CNumber("0.366014992+0.082669056i"), new CNumber("0.722726496+0.138715328i"), new CNumber("0.347126336+0.19244377599999998i"), new CNumber("0.491515504+0.55233424i"), new CNumber("0.066886928+0.6666319359999999i")},
                {new CNumber("0.3990462634+0.6740225322i"), new CNumber("0.3357015834+0.8261886916i"), new CNumber("0.6059490035999999+0.1858068608i"), new CNumber("0.36223140270000004+0.7380203122i"), new CNumber("0.7763801555000001+0.8478281344999999i"), new CNumber("0.3626558873+0.3713329124i"), new CNumber("0.41215483379999995+0.8282763326i"), new CNumber("0.5233028086+0.6430407116999999i"), new CNumber("0.5186529777+0.34008104120000004i"), new CNumber("0.5629394055+0.47584889049999995i"), new CNumber("0.5106912841+0.5746211495i")},
                {new CNumber("1.240943638+1.1585232890000001i"), new CNumber("1.065296188+0.6966004100000001i"), new CNumber("0.39530426100000005+1.190327061i"), new CNumber("0.7399437870000001+1.017628161i"), new CNumber("0.462715033+0.21286359600000002i"), new CNumber("0.545837215+0.5517095870000001i"), new CNumber("1.0194871829999999+0.692172965i"), new CNumber("0.738884278+0.7335727740000001i"), new CNumber("0.779787404+0.875736378i"), new CNumber("0.9683249840000001+0.9968224749999999i"), new CNumber("0.8921111100000001+0.578461433i")},
                {new CNumber("0.342094523+0.5679985527i"), new CNumber("0.3001019428+0.5753393303000001i"), new CNumber("0.1376240233+0.4412732301i"), new CNumber("0.6308110959+0.3023043986i"), new CNumber("0.6268204039999999+0.224719548i"), new CNumber("0.3259595543+0.21709579659999997i"), new CNumber("0.4552339839+0.2390891698i"), new CNumber("0.215841312+0.298389844i"), new CNumber("0.44761062189999995+0.18186170420000003i"), new CNumber("0.4662145364+0.3770295175i"), new CNumber("0.3209510801+0.1349159749i")},
                {new CNumber("0.15595753110000002+0.0354188919i"), new CNumber("0.0874513038+0.0708170169i"), new CNumber("0.1760693298+0.1715176029i"), new CNumber("0.0604864281+0.0756689199i"), new CNumber("0.1290927141+0.09166320870000001i"), new CNumber("0.028603572900000002+0.15409228590000001i"), new CNumber("0.18210117030000003+0.0626669526i"), new CNumber("0.05600266560000001+0.1520627934i"), new CNumber("0.14572888890000002+0.11588685360000002i"), new CNumber("0.053844795900000006+0.0380619519i"), new CNumber("0.05738460840000001+0.1663126626i")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")}};
        exp = new CMatrix(expEntries);

        assertTrue(exp.allClose(a.multTranspose(b)));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.25233, 0.97708, 0.63175, 0.41755};
        aRowIndices = new int[]{0, 1, 3, 4};
        aColIndices = new int[]{0, 0, 1, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[][]{
                {new CNumber("0.89404+0.48317i"), new CNumber("0.65646+0.8944i"), new CNumber("0.18768+0.19501i")}};
        b = new CMatrix(bEntries);

        expEntries = new CNumber[][]{
                {new CNumber("0.22559311319999997+0.1219182861i")},
                {new CNumber("0.8735486031999999+0.47209574359999995i")},
                {new CNumber("0.0")},
                {new CNumber("0.4147186050000001+0.5650372i")},
                {new CNumber("0.274104873+0.37345671999999996i")}};
        exp = new CMatrix(expEntries);

        assertTrue(exp.allClose(a.multTranspose(b)));

        // ---------------------  Sub-case 4 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.34473, 0.21113, 0.43909, 0.35121};
        aRowIndices = new int[]{2, 2, 4, 4};
        aColIndices = new int[]{1, 2, 1, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[][]{
                {new CNumber("0.75998+0.44056i"), new CNumber("0.53545+0.80806i"), new CNumber("0.22607+0.64841i")},
                {new CNumber("0.37701+0.88098i"), new CNumber("0.28264+0.49592i"), new CNumber("0.33644+0.421i")}};
        b = new CMatrix(bEntries);

        expEntries = new CNumber[][]{
                {new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.2323158376+0.41546132709999994i"), new CNumber("0.16846706439999998+0.25984423160000003i")},
                {new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.3145087852+0.5825391415i"), new CNumber("0.24226549000000003+0.3656129228i")}};
        exp = new CMatrix(expEntries);

        assertTrue(exp.allClose(a.multTranspose(b)));

        // ---------------------  Sub-case 5 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.82471, 0.446, 0.49633, 0.1568};
        aRowIndices = new int[]{0, 1, 2, 4};
        aColIndices = new int[]{0, 0, 0, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[][]{
                {new CNumber("0.43608+0.8293i"), new CNumber("0.49771+0.17778i")},
                {new CNumber("0.99398+0.60529i"), new CNumber("0.95101+0.41515i")},
                {new CNumber("0.34628+0.11562i"), new CNumber("0.73608+0.61967i")},
                {new CNumber("0.70272+0.91647i"), new CNumber("0.51192+0.58706i")},
                {new CNumber("0.27563+0.15206i"), new CNumber("0.87002+0.98647i")}};
        b = new CMatrix(bEntries);

        CooMatrix final0a = a;
        CMatrix final0b = b;
        assertThrows(Exception.class, ()->final0a.multTranspose(final0b));

        // ---------------------  Sub-case 6 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.12344, 0.51989, 0.0216, 0.3327};
        aRowIndices = new int[]{0, 0, 1, 1};
        aColIndices = new int[]{0, 1, 0, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[][]{
                {new CNumber("0.09289+0.53643i"), new CNumber("0.37934+0.37787i"), new CNumber("0.62+0.63027i"), new CNumber("0.23256+0.47007i"), new CNumber("0.09347+0.03001i")},
                {new CNumber("0.31459+0.25663i"), new CNumber("0.31468+0.39577i"), new CNumber("0.17622+0.45597i"), new CNumber("0.63123+0.20856i"), new CNumber("0.30242+0.85175i")},
                {new CNumber("0.00654+0.57839i"), new CNumber("0.36914+0.6587i"), new CNumber("0.18079+0.49582i"), new CNumber("0.0501+0.26054i"), new CNumber("0.20055+0.18699i")},
                {new CNumber("0.21655+0.57022i"), new CNumber("0.38342+0.44403i"), new CNumber("0.94312+0.93926i"), new CNumber("0.9012+0.53264i"), new CNumber("0.13753+0.49135i")},
                {new CNumber("0.67475+0.10138i"), new CNumber("0.7811+0.90816i"), new CNumber("0.14068+0.5877i"), new CNumber("0.90754+0.07487i"), new CNumber("0.43711+0.44065i")}};
        b = new CMatrix(bEntries);

        CooMatrix final1a = a;
        CMatrix final1b = b;
        assertThrows(Exception.class, ()->final1a.multTranspose(final1b));
    }
}
