package org.flag4j.sparse_matrix;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.sparse.CooCMatrixOld;
import org.flag4j.arrays_old.sparse.CooMatrixOld;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooMatrixElemMultTests {
    Shape aShape, bShape, expShape;
    double[] aEntries;
    CooMatrixOld A;
    int[] aRowIndices, aColIndices, bRowIndices, bColIndices, expRowIndices, expColIndices;

    @Test
    void realSparseRealSparseElem_multTest() {
        double[] bEntries;
        CooMatrixOld B;
        double[] expEntries;
        CooMatrixOld exp;

        // ------------------- Sub-case 1 -------------------
        aShape = new Shape(5, 5);
        aEntries = new double[]{0.866448287823294, 0.30737293330910287, 0.06271913235416882, 0.9462772376882058, 0.5531203613914151};
        aRowIndices = new int[]{0, 1, 2, 3, 3};
        aColIndices = new int[]{0, 1, 0, 2, 3};
        A = new CooMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 5);
        bEntries = new double[]{0.36671915635199404, 0.3428698034980757, 0.9097041832944237, 0.11428172868423914, 0.7440256197049736};
        bRowIndices = new int[]{2, 2, 3, 3, 4};
        bColIndices = new int[]{2, 4, 0, 1, 1};
        B = new CooMatrixOld(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(5, 5);
        expEntries = new double[]{};
        expRowIndices = new int[]{};
        expColIndices = new int[]{};
        exp = new CooMatrixOld(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, A.elemMult(B));

        // ------------------- Sub-case 2 -------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.28882178224257016, 0.49765654175915963, 0.280286313350515};
        aRowIndices = new int[]{1, 2, 2};
        aColIndices = new int[]{4, 3, 4};
        A = new CooMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3, 5);
        bEntries = new double[]{0.34868159264004794, 0.5423192572450884, 0.5731826048442968};
        bRowIndices = new int[]{0, 0, 1};
        bColIndices = new int[]{0, 4, 1};
        B = new CooMatrixOld(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(3, 5);
        expEntries = new double[]{};
        expRowIndices = new int[]{};
        expColIndices = new int[]{};
        exp = new CooMatrixOld(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, A.elemMult(B));

        // ------------------- Sub-case 3 -------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.5126288650472578, 0.5012195332635256, 0.678674045250447};
        aRowIndices = new int[]{0, 0, 1};
        aColIndices = new int[]{1, 3, 1};
        A = new CooMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 3);
        bEntries = new double[]{0.691833293743732, 0.47552961961396856, 0.40668901076853314};
        bRowIndices = new int[]{0, 1, 4};
        bColIndices = new int[]{2, 1, 2};
        B = new CooMatrixOld(bShape, bEntries, bRowIndices, bColIndices);

        CooMatrixOld finalB = B;
        assertThrows(Exception.class, ()->A.elemMult(finalB));
    }


    @Test
    void realSparseComplexSparseElem_multTest() {
        CNumber[] bEntries;
        CooCMatrixOld B;
        CNumber[] expEntries;
        CooCMatrixOld exp;

        // ------------------- Sub-case 1 -------------------
        aShape = new Shape(5, 5);
        aEntries = new double[]{0.4774939644223758, 0.7246896629788531, 0.3839537072633754, 0.651469248100361, 0.9812531920891868};
        aRowIndices = new int[]{0, 1, 2, 2, 4};
        aColIndices = new int[]{4, 2, 1, 2, 4};
        A = new CooMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 5);
        bEntries = new CNumber[]{new CNumber("0.1149124855929522+0.6418603333266117i"), new CNumber("0.35226526921203327+0.24572457998610397i"), new CNumber("0.8851769121217483+0.9209054367586471i"), new CNumber("0.9192389886528087+0.9075131438657206i"), new CNumber("0.8502078577629006+0.8236183481807278i")};
        bRowIndices = new int[]{0, 1, 2, 2, 3};
        bColIndices = new int[]{2, 4, 0, 2, 2};
        B = new CooCMatrixOld(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(5, 5);
        expEntries = new CNumber[]{new CNumber("0.5988559327621816+0.5912169054753957i")};
        expRowIndices = new int[]{2};
        expColIndices = new int[]{2};
        exp = new CooCMatrixOld(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, A.elemMult(B));

        // ------------------- Sub-case 2 -------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.14966329052090532, 0.07461663082197267, 0.9978836866960403};
        aRowIndices = new int[]{0, 1, 1};
        aColIndices = new int[]{2, 3, 4};
        A = new CooMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3, 5);
        bEntries = new CNumber[]{new CNumber("0.349098415330798+0.321942254137234i"), new CNumber("0.7854043222570336+0.22540524127991513i"), new CNumber("0.8011525456939302+0.9321632652950249i")};
        bRowIndices = new int[]{1, 2, 2};
        bColIndices = new int[]{4, 0, 4};
        B = new CooCMatrixOld(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(3, 5);
        expEntries = new CNumber[]{new CNumber("0.3483596137100422+0.3212609234616966i")};
        expRowIndices = new int[]{1};
        expColIndices = new int[]{4};
        exp = new CooCMatrixOld(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, A.elemMult(B));

        // ------------------- Sub-case 3 -------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.1322731985701755, 0.9085559526923129, 0.5431039394927907};
        aRowIndices = new int[]{0, 0, 1};
        aColIndices = new int[]{3, 4, 0};
        A = new CooMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 3);
        bEntries = new CNumber[]{new CNumber("0.9435757560157966+0.6397816152125482i"), new CNumber("0.7179825060253362+0.9107097243058598i"), new CNumber("0.7821907013282112+0.18834383798665189i")};
        bRowIndices = new int[]{1, 3, 4};
        bColIndices = new int[]{0, 0, 2};
        B = new CooCMatrixOld(bShape, bEntries, bRowIndices, bColIndices);

        CooCMatrixOld finalB = B;
        assertThrows(Exception.class, ()->A.elemMult(finalB));
    }


    @Test
    void realSparseRealDenseElem_multTest() {
        double[][] bEntries;
        MatrixOld B;
        double[] expEntries;
        CooMatrixOld exp;

        // ------------------- Sub-case 1 -------------------
        aShape = new Shape(5, 5);
        aEntries = new double[]{0.9266615654453623, 0.9955743042782697, 0.35786926201846503, 0.5855592205067941, 0.7318504682591063};
        aRowIndices = new int[]{1, 2, 3, 4, 4};
        aColIndices = new int[]{1, 3, 3, 3, 4};
        A = new CooMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.55042, 0.09573, 0.61282, 0.62434, 0.67614},
                {0.28339, 0.42644, 0.68006, 0.65665, 0.57716},
                {0.1891, 0.04051, 0.97136, 0.84744, 0.94883},
                {0.85845, 0.69155, 0.79853, 0.91054, 0.90991},
                {0.59693, 0.84758, 0.95484, 0.87886, 0.18658}};
        B = new MatrixOld(bEntries);

        expShape = new Shape(5, 5);
        expEntries = new double[]{0.39516555796852026, 0.8436894884175768, 0.3258542778382931, 0.514624576534601, 0.13654866036778404};
        expRowIndices = new int[]{1, 2, 3, 4, 4};
        expColIndices = new int[]{1, 3, 3, 3, 4};
        exp = new CooMatrixOld(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, A.elemMult(B));

        // ------------------- Sub-case 2 -------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.18977752443207252, 0.8975104457922927, 0.6221398237590096};
        aRowIndices = new int[]{0, 1, 2};
        aColIndices = new int[]{0, 0, 3};
        A = new CooMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.17961, 0.27749, 0.56096, 0.15615, 0.42525},
                {0.96071, 0.36572, 0.35921, 0.79275, 0.98678},
                {0.07837, 0.31829, 0.22698, 0.87646, 0.46729}};
        B = new MatrixOld(bEntries);

        expShape = new Shape(3, 5);
        expEntries = new double[]{0.034085941163244544, 0.8622472603771134, 0.5452806699318216};
        expRowIndices = new int[]{0, 1, 2};
        expColIndices = new int[]{0, 0, 3};
        exp = new CooMatrixOld(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, A.elemMult(B));

        // ------------------- Sub-case 3 -------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.4687563393386337, 0.7171208360117912, 0.29932279969931574};
        aRowIndices = new int[]{0, 1, 1};
        aColIndices = new int[]{3, 3, 4};
        A = new CooMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.76405, 0.03985, 0.51432},
                {0.52806, 0.19013, 0.42413},
                {0.26754, 0.41276, 0.27386},
                {0.2249, 0.39213, 0.6063},
                {0.19165, 0.72628, 0.9815}};
        B = new MatrixOld(bEntries);

        MatrixOld finalB = B;
        assertThrows(Exception.class, ()->A.elemMult(finalB));
    }


    @Test
    void realSparseComplexDenseElem_multTest() {
        CNumber[][] bEntries;
        CMatrixOld B;
        CNumber[] expEntries;
        CooCMatrixOld exp;

        // ------------------- Sub-case 1 -------------------
        aShape = new Shape(5, 5);
        aEntries = new double[]{0.8675343800623778, 0.9418895620171525, 0.4483662521734234, 0.955802004625131, 0.35124297521347314};
        aRowIndices = new int[]{0, 0, 2, 3, 4};
        aColIndices = new int[]{0, 2, 4, 0, 4};
        A = new CooMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[][]{
                {new CNumber("0.14548+0.3265i"), new CNumber("0.27441+0.69396i"), new CNumber("0.73925+0.28742i"), new CNumber("0.62831+0.11888i"), new CNumber("0.8787+0.50604i")},
                {new CNumber("0.63723+0.842i"), new CNumber("0.76641+0.19674i"), new CNumber("0.43291+0.22406i"), new CNumber("0.67014+0.11522i"), new CNumber("0.33123+0.82666i")},
                {new CNumber("0.86974+0.50151i"), new CNumber("0.22129+0.75143i"), new CNumber("0.64144+0.01722i"), new CNumber("0.01703+0.81511i"), new CNumber("0.45808+0.67284i")},
                {new CNumber("0.40227+0.96917i"), new CNumber("0.76246+0.00764i"), new CNumber("0.50002+0.31821i"), new CNumber("0.90325+0.82669i"), new CNumber("0.91393+0.54211i")},
                {new CNumber("0.32864+0.12926i"), new CNumber("0.74844+0.9638i"), new CNumber("0.22918+0.38571i"), new CNumber("0.23339+0.18911i"), new CNumber("0.19796+0.10059i")}};
        B = new CMatrixOld(bEntries);

        expShape = new Shape(5, 5);
        expEntries = new CNumber[]{new CNumber("0.12620890161147472+0.28324997509036637i"), new CNumber("0.6962918587211799+0.27071789791497i"), new CNumber("0.20538761279560178+0.3016787491123662i"), new CNumber("0.38449047240055145+0.9263346288225381i"), new CNumber("0.06953205937325914+0.03533153087672326i")};
        expRowIndices = new int[]{0, 0, 2, 3, 4};
        expColIndices = new int[]{0, 2, 4, 0, 4};
        exp = new CooCMatrixOld(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, A.elemMult(B));

        // ------------------- Sub-case 2 -------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.8256031726855145, 0.7269797668376734, 0.29324283281791996};
        aRowIndices = new int[]{0, 1, 2};
        aColIndices = new int[]{3, 0, 3};
        A = new CooMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[][]{
                {new CNumber("0.98119+0.15994i"), new CNumber("0.06758+0.23465i"), new CNumber("0.17934+0.38625i"), new CNumber("0.76932+0.48628i"), new CNumber("0.797+0.18852i")},
                {new CNumber("0.06161+0.75494i"), new CNumber("0.71145+0.91536i"), new CNumber("0.32839+0.93935i"), new CNumber("0.95857+0.02112i"), new CNumber("0.89824+0.18991i")},
                {new CNumber("0.68639+0.57294i"), new CNumber("0.29695+0.9318i"), new CNumber("0.35713+0.87581i"), new CNumber("0.90308+0.60565i"), new CNumber("0.78263+0.49965i")}};
        B = new CMatrixOld(bEntries);

        expShape = new Shape(3, 5);
        expEntries = new CNumber[]{new CNumber("0.63515303281042+0.401474310813512i"), new CNumber("0.044789223434869053+0.5488261051764332i"), new CNumber("0.2648217374612072+0.17760252169617324i")};
        expRowIndices = new int[]{0, 1, 2};
        expColIndices = new int[]{3, 0, 3};
        exp = new CooCMatrixOld(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, A.elemMult(B));

        // ------------------- Sub-case 3 -------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.7994673821482825, 0.3451259014862351, 0.7124467804540862};
        aRowIndices = new int[]{0, 1, 2};
        aColIndices = new int[]{4, 3, 2};
        A = new CooMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[][]{
                {new CNumber("0.08096+0.20643i"), new CNumber("0.80319+0.69336i"), new CNumber("0.85421+0.68421i")},
                {new CNumber("0.57188+0.81598i"), new CNumber("0.83954+0.95383i"), new CNumber("0.3958+0.24863i")},
                {new CNumber("0.06085+0.36061i"), new CNumber("0.65606+0.59434i"), new CNumber("0.38108+0.18092i")},
                {new CNumber("0.68197+0.4751i"), new CNumber("0.59857+0.1467i"), new CNumber("0.25484+0.51415i")},
                {new CNumber("0.93641+0.63461i"), new CNumber("0.4401+0.88435i"), new CNumber("0.61362+0.21701i")}};
        B = new CMatrixOld(bEntries);

        CMatrixOld finalB = B;
        assertThrows(Exception.class, ()->A.elemMult(finalB));
    }
}
