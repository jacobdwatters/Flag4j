package org.flag4j.sparse_matrix;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.sparse.CooCMatrixOld;
import org.flag4j.arrays_old.sparse.CooMatrixOld;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.junit.jupiter.api.Test;

import static org.junit.Assert.assertThrows;
import static org.junit.jupiter.api.Assertions.assertEquals;

class CooMatrixStackTests {

    @Test
    void realSparseStackTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrixOld a;

        Shape bShape;
        int[] bRowIndices;
        int[] bColIndices;
        double[] bEntries;
        CooMatrixOld b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        double[] expEntries;
        CooMatrixOld exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(2, 3);
        aEntries = new double[]{0.6994};
        aRowIndices = new int[]{0};
        aColIndices = new int[]{2};
        a = new CooMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(4, 3);
        bEntries = new double[]{0.53334, 0.36866};
        bRowIndices = new int[]{1, 3};
        bColIndices = new int[]{1, 1};
        b = new CooMatrixOld(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(6, 3);
        expEntries = new double[]{0.6994, 0.53334, 0.36866};
        expRowIndices = new int[]{0, 3, 5};
        expColIndices = new int[]{2, 1, 1};
        exp = new CooMatrixOld(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.stack(b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(1, 2);
        aEntries = new double[]{};
        aRowIndices = new int[]{};
        aColIndices = new int[]{};
        a = new CooMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 2);
        bEntries = new double[]{0.54824, 0.3091};
        bRowIndices = new int[]{2, 4};
        bColIndices = new int[]{0, 1};
        b = new CooMatrixOld(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(6, 2);
        expEntries = new double[]{0.54824, 0.3091};
        expRowIndices = new int[]{3, 5};
        expColIndices = new int[]{0, 1};
        exp = new CooMatrixOld(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.stack(b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(14, 5);
        aEntries = new double[]{0.95223, 0.17095, 0.32156, 0.9396, 0.3762, 0.10813, 0.08055, 0.31545, 0.15498, 0.91676, 0.83886, 0.94144, 0.30784, 0.49851};
        aRowIndices = new int[]{0, 0, 1, 4, 4, 4, 5, 5, 6, 6, 6, 9, 10, 11};
        aColIndices = new int[]{0, 2, 0, 2, 3, 4, 1, 3, 0, 2, 4, 3, 0, 0};
        a = new CooMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(14, 6);
        bEntries = new double[]{0.05335, 0.11684, 0.88899, 0.80113, 0.27065, 0.60548, 0.61726, 0.29856, 0.96785, 0.66154, 0.07842, 0.44824, 0.06887, 0.53813, 0.72747, 0.93312, 0.04432};
        bRowIndices = new int[]{0, 0, 0, 0, 2, 2, 2, 4, 4, 6, 7, 7, 8, 9, 9, 10, 13};
        bColIndices = new int[]{1, 2, 4, 5, 1, 3, 5, 4, 5, 1, 1, 5, 2, 4, 5, 5, 2};
        b = new CooMatrixOld(bShape, bEntries, bRowIndices, bColIndices);

        CooMatrixOld finala = a;
        CooMatrixOld finalb = b;
        assertThrows(Exception.class, ()->finala.stack(finalb));
    }


    @Test
    void complexSparseStackTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrixOld a;

        Shape bShape;
        int[] bRowIndices;
        int[] bColIndices;
        CNumber[] bEntries;
        CooCMatrixOld b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        CooCMatrixOld exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(2, 3);
        aEntries = new double[]{0.71486};
        aRowIndices = new int[]{1};
        aColIndices = new int[]{0};
        a = new CooMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(4, 3);
        bEntries = new CNumber[]{new CNumber("0.41503+0.75531i"), new CNumber("0.71333+0.30431i")};
        bRowIndices = new int[]{1, 2};
        bColIndices = new int[]{0, 2};
        b = new CooCMatrixOld(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(6, 3);
        expEntries = new CNumber[]{new CNumber("0.71486"), new CNumber("0.41503+0.75531i"), new CNumber("0.71333+0.30431i")};
        expRowIndices = new int[]{1, 3, 4};
        expColIndices = new int[]{0, 0, 2};
        exp = new CooCMatrixOld(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.stack(b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(1, 2);
        aEntries = new double[]{};
        aRowIndices = new int[]{};
        aColIndices = new int[]{};
        a = new CooMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 2);
        bEntries = new CNumber[]{new CNumber("0.35649+0.27366i"), new CNumber("0.71799+0.64388i")};
        bRowIndices = new int[]{0, 2};
        bColIndices = new int[]{1, 0};
        b = new CooCMatrixOld(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(6, 2);
        expEntries = new CNumber[]{new CNumber("0.35649+0.27366i"), new CNumber("0.71799+0.64388i")};
        expRowIndices = new int[]{1, 3};
        expColIndices = new int[]{1, 0};
        exp = new CooCMatrixOld(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.stack(b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(14, 5);
        aEntries = new double[]{0.72813, 0.9915, 0.03256, 0.96153, 0.82032, 0.27925, 0.59001, 0.5632, 0.83047, 0.80622, 0.74926, 0.53789, 0.89853, 0.87996};
        aRowIndices = new int[]{0, 0, 1, 2, 2, 2, 3, 3, 4, 5, 6, 8, 12, 13};
        aColIndices = new int[]{0, 4, 0, 0, 1, 3, 0, 4, 4, 4, 0, 0, 4, 1};
        a = new CooMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(14, 6);
        bEntries = new CNumber[]{new CNumber("0.21418+0.77678i"), new CNumber("0.33767+0.04435i"), new CNumber("0.19049+0.17144i"), new CNumber("0.57794+0.43139i"), new CNumber("0.90346+0.20636i"), new CNumber("0.50151+0.63914i"), new CNumber("0.15893+0.10915i"), new CNumber("0.71894+0.00546i"), new CNumber("0.75743+0.09909i"), new CNumber("0.59119+0.38895i"), new CNumber("0.06088+0.00477i"), new CNumber("0.63655+0.63256i"), new CNumber("0.09156+0.54102i"), new CNumber("0.73534+0.75135i"), new CNumber("0.30838+0.43967i"), new CNumber("0.03917+0.93622i"), new CNumber("0.07129+0.87944i")};
        bRowIndices = new int[]{2, 2, 3, 5, 6, 6, 6, 9, 9, 10, 10, 11, 11, 11, 12, 12, 12};
        bColIndices = new int[]{4, 5, 5, 3, 2, 3, 5, 0, 1, 2, 4, 2, 3, 5, 2, 3, 4};
        b = new CooCMatrixOld(bShape, bEntries, bRowIndices, bColIndices);

        CooMatrixOld finala = a;
        CooCMatrixOld finalb = b;
        assertThrows(Exception.class, ()->finala.stack(finalb));
    }


    @Test
    void realDenseStackTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrixOld a;

        double[][] bEntries;
        MatrixOld b;

        double[][] expEntries;
        MatrixOld exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(2, 3);
        aEntries = new double[]{0.42414};
        aRowIndices = new int[]{0};
        aColIndices = new int[]{0};
        a = new CooMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.42346, 0.45166, 0.29951},
                {0.86687, 0.83652, 0.22394},
                {0.51345, 0.72834, 0.90933},
                {0.72956, 0.68369, 0.53742}};
        b = new MatrixOld(bEntries);

        expEntries = new double[][]{
                {0.42414, 0.0, 0.0},
                {0.0, 0.0, 0.0},
                {0.42346, 0.45166, 0.29951},
                {0.86687, 0.83652, 0.22394},
                {0.51345, 0.72834, 0.90933},
                {0.72956, 0.68369, 0.53742}};
        exp = new MatrixOld(expEntries);

        assertEquals(exp, a.stack(b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(1, 2);
        aEntries = new double[]{};
        aRowIndices = new int[]{};
        aColIndices = new int[]{};
        a = new CooMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.03722, 0.0959},
                {0.82004, 0.79942},
                {0.85848, 0.92187},
                {0.87077, 0.43443},
                {0.1931, 0.41343}};
        b = new MatrixOld(bEntries);

        expEntries = new double[][]{
                {0.0, 0.0},
                {0.03722, 0.0959},
                {0.82004, 0.79942},
                {0.85848, 0.92187},
                {0.87077, 0.43443},
                {0.1931, 0.41343}};
        exp = new MatrixOld(expEntries);

        assertEquals(exp, a.stack(b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(14, 5);
        aEntries = new double[]{0.29338, 0.93129, 0.40606, 0.13409, 0.83574, 0.6591, 0.09212, 0.87431, 0.70901, 0.42248, 0.77734, 0.32068, 0.8386, 0.87708};
        aRowIndices = new int[]{0, 0, 1, 1, 2, 4, 4, 5, 5, 5, 9, 10, 11, 13};
        aColIndices = new int[]{0, 1, 0, 1, 3, 2, 4, 0, 1, 3, 1, 1, 3, 0};
        a = new CooMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.75836, 0.00024, 0.25097, 0.06717, 0.87113, 0.20188},
                {0.63185, 0.70242, 0.50114, 0.49862, 0.56076, 0.85131},
                {0.77312, 0.05837, 0.6626, 0.33361, 0.78283, 0.92395},
                {0.84929, 0.50391, 0.0196, 0.9233, 0.87276, 0.95599},
                {0.22842, 0.65994, 0.40556, 0.07594, 0.83461, 0.26268},
                {0.51504, 0.31063, 0.31369, 0.39844, 0.88845, 0.29644},
                {0.41285, 0.80511, 0.37801, 0.35051, 0.65639, 0.05778},
                {0.18765, 0.9244, 0.03848, 0.76052, 0.57481, 0.20424},
                {0.71705, 0.31667, 0.54783, 0.30136, 0.61207, 0.00745},
                {0.08881, 0.60552, 0.45187, 0.28724, 0.62118, 0.28671},
                {0.83733, 0.49654, 0.37077, 0.9125, 0.74755, 0.48363},
                {0.0529, 0.6966, 0.65055, 0.25113, 0.79863, 0.81757},
                {0.54318, 0.0829, 0.68785, 0.58007, 0.98927, 0.11201},
                {0.97607, 0.5196, 0.44965, 0.89517, 0.93744, 0.03956}};
        b = new MatrixOld(bEntries);

        CooMatrixOld finala = a;
        MatrixOld finalb = b;
        assertThrows(Exception.class, ()->finala.stack(finalb));
    }


    @Test
    void complexDenseStackTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrixOld a;

        CNumber[][] bEntries;
        CMatrixOld b;

        CNumber[][] expEntries;
        CMatrixOld exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(2, 3);
        aEntries = new double[]{0.08155};
        aRowIndices = new int[]{0};
        aColIndices = new int[]{0};
        a = new CooMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[][]{
                {new CNumber("0.84939+0.31005i"), new CNumber("0.62801+0.8259i"), new CNumber("0.05949+0.72899i")},
                {new CNumber("0.73232+0.80282i"), new CNumber("0.51932+0.96624i"), new CNumber("0.08145+0.29743i")},
                {new CNumber("0.74275+0.87904i"), new CNumber("0.2309+0.97304i"), new CNumber("0.88203+0.44979i")},
                {new CNumber("0.34869+0.73754i"), new CNumber("0.89527+0.90273i"), new CNumber("0.14601+0.11972i")}};
        b = new CMatrixOld(bEntries);

        expEntries = new CNumber[][]{
                {new CNumber("0.08155"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.84939+0.31005i"), new CNumber("0.62801+0.8259i"), new CNumber("0.05949+0.72899i")},
                {new CNumber("0.73232+0.80282i"), new CNumber("0.51932+0.96624i"), new CNumber("0.08145+0.29743i")},
                {new CNumber("0.74275+0.87904i"), new CNumber("0.2309+0.97304i"), new CNumber("0.88203+0.44979i")},
                {new CNumber("0.34869+0.73754i"), new CNumber("0.89527+0.90273i"), new CNumber("0.14601+0.11972i")}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, a.stack(b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(1, 2);
        aEntries = new double[]{};
        aRowIndices = new int[]{};
        aColIndices = new int[]{};
        a = new CooMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[][]{
                {new CNumber("0.73431+0.59055i"), new CNumber("0.7604+0.03804i")},
                {new CNumber("0.78981+0.98193i"), new CNumber("0.95475+0.1637i")},
                {new CNumber("0.07054+0.09212i"), new CNumber("0.81147+0.13819i")},
                {new CNumber("0.75877+0.73089i"), new CNumber("0.82998+0.90563i")},
                {new CNumber("0.17084+0.47144i"), new CNumber("0.87103+0.10453i")}};
        b = new CMatrixOld(bEntries);

        expEntries = new CNumber[][]{
                {new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.73431+0.59055i"), new CNumber("0.7604+0.03804i")},
                {new CNumber("0.78981+0.98193i"), new CNumber("0.95475+0.1637i")},
                {new CNumber("0.07054+0.09212i"), new CNumber("0.81147+0.13819i")},
                {new CNumber("0.75877+0.73089i"), new CNumber("0.82998+0.90563i")},
                {new CNumber("0.17084+0.47144i"), new CNumber("0.87103+0.10453i")}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, a.stack(b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(14, 5);
        aEntries = new double[]{0.22748, 0.15823, 0.01155, 0.72994, 0.42505, 0.6026, 0.46703, 0.823, 0.1616, 0.8939, 0.16783, 0.06605, 0.21062, 0.72352};
        aRowIndices = new int[]{1, 2, 4, 5, 5, 6, 7, 8, 9, 9, 11, 11, 12, 13};
        aColIndices = new int[]{0, 4, 0, 2, 4, 4, 0, 1, 2, 3, 0, 4, 2, 2};
        a = new CooMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[][]{
                {new CNumber("0.1484+0.81772i"), new CNumber("0.42885+0.60199i"), new CNumber("0.80477+0.62236i"), new CNumber("0.79025+0.12373i"), new CNumber("0.92617+0.74118i"), new CNumber("0.30808+0.61041i")},
                {new CNumber("0.94671+0.77263i"), new CNumber("0.81093+0.9994i"), new CNumber("0.80472+0.16465i"), new CNumber("0.09977+0.41369i"), new CNumber("0.52559+0.39216i"), new CNumber("0.85602+0.09586i")},
                {new CNumber("0.49482+0.47204i"), new CNumber("0.89238+0.07097i"), new CNumber("0.11517+0.82619i"), new CNumber("0.27601+0.50809i"), new CNumber("0.94994+0.89756i"), new CNumber("0.38582+0.61161i")},
                {new CNumber("0.04157+0.97631i"), new CNumber("0.64102+0.78834i"), new CNumber("0.69379+0.74135i"), new CNumber("0.36546+0.57254i"), new CNumber("0.92969+0.84928i"), new CNumber("0.41364+0.564i")},
                {new CNumber("0.72206+0.904i"), new CNumber("0.01353+0.75145i"), new CNumber("0.61068+0.50192i"), new CNumber("0.88384+0.44049i"), new CNumber("0.64612+0.55604i"), new CNumber("0.61198+0.12695i")},
                {new CNumber("0.87571+0.92566i"), new CNumber("0.0769+0.10847i"), new CNumber("0.03763+0.73221i"), new CNumber("0.53073+0.09559i"), new CNumber("0.50231+0.32042i"), new CNumber("0.47476+0.99153i")},
                {new CNumber("0.10405+0.14869i"), new CNumber("0.28998+0.64539i"), new CNumber("0.41649+0.14627i"), new CNumber("0.29615+0.78516i"), new CNumber("0.46753+0.13839i"), new CNumber("0.5605+0.58083i")},
                {new CNumber("0.27978+0.75316i"), new CNumber("0.09644+0.38716i"), new CNumber("0.07721+0.43908i"), new CNumber("0.15135+0.20482i"), new CNumber("0.35144+0.30357i"), new CNumber("0.17262+0.21103i")},
                {new CNumber("0.24817+0.68733i"), new CNumber("0.9175+0.35786i"), new CNumber("0.38858+0.95425i"), new CNumber("0.88688+0.31407i"), new CNumber("0.54171+0.85257i"), new CNumber("0.23647+0.25302i")},
                {new CNumber("0.32837+0.00633i"), new CNumber("0.12228+0.27998i"), new CNumber("0.72095+0.90433i"), new CNumber("0.55664+0.30704i"), new CNumber("0.96143+0.42851i"), new CNumber("0.50477+0.44901i")},
                {new CNumber("0.49495+0.47243i"), new CNumber("0.0359+0.10143i"), new CNumber("0.03746+0.43852i"), new CNumber("0.11778+0.77948i"), new CNumber("0.8621+0.20514i"), new CNumber("0.06194+0.69523i")},
                {new CNumber("0.71155+0.88352i"), new CNumber("0.46062+0.12308i"), new CNumber("0.29058+0.66116i"), new CNumber("0.66899+0.92729i"), new CNumber("0.79081+0.98986i"), new CNumber("0.95555+0.40087i")},
                {new CNumber("0.63158+0.95641i"), new CNumber("0.56604+0.93587i"), new CNumber("0.49765+0.25932i"), new CNumber("0.83888+0.18245i"), new CNumber("0.65702+0.20741i"), new CNumber("0.81914+0.38574i")},
                {new CNumber("0.89991+0.27551i"), new CNumber("0.26809+0.75698i"), new CNumber("0.34166+0.10185i"), new CNumber("0.06181+0.35822i"), new CNumber("0.38711+0.73425i"), new CNumber("0.74133+0.29344i")}};
        b = new CMatrixOld(bEntries);

        CooMatrixOld finala = a;
        CMatrixOld finalb = b;
        assertThrows(Exception.class, ()->finala.stack(finalb));
    }
}
