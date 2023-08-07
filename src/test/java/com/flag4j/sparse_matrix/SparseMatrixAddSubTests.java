package com.flag4j.sparse_matrix;

import com.flag4j.*;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class SparseMatrixAddSubTests {
    Shape aShape;
    double[] aEntries;
    int[] aRowIndices, aColIndices;
    SparseMatrix A;

    Shape bShape, expShape;
    int[] bRowIndices, bColIndices, expRowIndices, expColIndices;

    @Test
    void realSparseRealSparseSubTest() {
        double[] bEntries;
        SparseMatrix B;
        double[] expEntries;
        SparseMatrix exp;

        // ------------------- Sub-case 1 -------------------
        aShape = new Shape(5, 5);
        aEntries = new double[]{0.7197167862026044, 0.8274741025737611, 0.44635157987459506, 0.9106722384653576, 0.11927378948791945};
        aRowIndices = new int[]{0, 1, 1, 2, 3};
        aColIndices = new int[]{0, 2, 4, 1, 4};
        A = new SparseMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 5);
        bEntries = new double[]{0.47157415615670795, 0.18857508525597733, 0.35596697752675244, 0.007607366738096366, 0.7964396954007252};
        bRowIndices = new int[]{0, 1, 1, 2, 4};
        bColIndices = new int[]{4, 2, 3, 4, 1};
        B = new SparseMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(5, 5);
        expEntries = new double[]{0.7197167862026044, -0.47157415615670795, 0.6388990173177838, -0.35596697752675244, 0.44635157987459506, 0.9106722384653576, -0.007607366738096366, 0.11927378948791945, -0.7964396954007252};
        expRowIndices = new int[]{0, 0, 1, 1, 1, 2, 2, 3, 4};
        expColIndices = new int[]{0, 4, 2, 3, 4, 1, 4, 4, 1};
        exp = new SparseMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, A.sub(B));

        // ------------------- Sub-case 2 -------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.7234897223335167, 0.4569611853869002, 0.17516090675041163};
        aRowIndices = new int[]{2, 2, 2};
        aColIndices = new int[]{0, 1, 2};
        A = new SparseMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3, 5);
        bEntries = new double[]{0.8234991857678657, 0.7336285038619206, 0.18279094243044902};
        bRowIndices = new int[]{1, 1, 2};
        bColIndices = new int[]{1, 2, 2};
        B = new SparseMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(3, 5);
        expEntries = new double[]{-0.8234991857678657, -0.7336285038619206, 0.7234897223335167, 0.4569611853869002, -0.007630035680037395};
        expRowIndices = new int[]{1, 1, 2, 2, 2};
        expColIndices = new int[]{1, 2, 0, 1, 2};
        exp = new SparseMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, A.sub(B));

        // ------------------- Sub-case 3 -------------------
        aShape = new Shape(9, 5);
        aEntries = new double[]{0.9180438547915879, 0.07235711960675528, 0.2657856824106707, 0.47790644045925923, 0.003070610781322758, 0.34202224059852215, 0.27031789435156306, 0.9499530477799731, 0.08321871287149174};
        aRowIndices = new int[]{0, 1, 3, 4, 6, 6, 6, 8, 8};
        aColIndices = new int[]{3, 0, 0, 1, 0, 2, 3, 1, 3};
        A = new SparseMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3, 5);
        bEntries = new double[]{0.646214900340821, 0.046463495109891007, 0.052648659293113576};
        bRowIndices = new int[]{0, 1, 2};
        bColIndices = new int[]{0, 2, 2};
        B = new SparseMatrix(bShape, bEntries, bRowIndices, bColIndices);

        SparseMatrix finalB = B;
        assertThrows(Exception.class, ()->A.sub(finalB));
    }


    @Test
    void realSparseComplexSparseSubTest() {
        CNumber[] bEntries;
        SparseCMatrix B;
        CNumber[] expEntries;
        SparseCMatrix exp;

        // ------------------- Sub-case 1 -------------------
        aShape = new Shape(5, 5);
        aEntries = new double[]{0.7540312001068363, 0.9428223651945967, 0.4650843608728411, 0.5206860140992323, 0.38382860306767685};
        aRowIndices = new int[]{0, 0, 1, 2, 3};
        aColIndices = new int[]{0, 2, 3, 0, 1};
        A = new SparseMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 5);
        bEntries = new CNumber[]{new CNumber("0.36149192006867104+0.04319130594448195i"), new CNumber("0.4360933807209263+0.7955125874980291i"), new CNumber("0.8429639560748449+0.867498720512465i"), new CNumber("0.016258069078200377+0.4119653895482479i"), new CNumber("0.9945335908417812+0.42006061053116317i")};
        bRowIndices = new int[]{1, 2, 2, 3, 4};
        bColIndices = new int[]{4, 0, 4, 1, 3};
        B = new SparseCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(5, 5);
        expEntries = new CNumber[]{new CNumber("0.7540312001068363"), new CNumber("0.9428223651945967"), new CNumber("0.4650843608728411"), new CNumber("-0.36149192006867104-0.04319130594448195i"), new CNumber("0.08459263337830603-0.7955125874980291i"), new CNumber("-0.8429639560748449-0.867498720512465i"), new CNumber("0.3675705339894765-0.4119653895482479i"), new CNumber("-0.9945335908417812-0.42006061053116317i")};
        expRowIndices = new int[]{0, 0, 1, 1, 2, 2, 3, 4};
        expColIndices = new int[]{0, 2, 3, 4, 0, 4, 1, 3};
        exp = new SparseCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, A.sub(B));

        // ------------------- Sub-case 2 -------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.4602784916680196, 0.6975585135512675, 0.2503474806442201};
        aRowIndices = new int[]{0, 0, 2};
        aColIndices = new int[]{1, 3, 0};
        A = new SparseMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3, 5);
        bEntries = new CNumber[]{new CNumber("0.02594980913117173+0.4379367199630513i"), new CNumber("0.8883365921139904+0.5585597123976269i"), new CNumber("0.5294253993237487+0.7455315910395643i")};
        bRowIndices = new int[]{0, 0, 1};
        bColIndices = new int[]{1, 3, 1};
        B = new SparseCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(3, 5);
        expEntries = new CNumber[]{new CNumber("0.43432868253684787-0.4379367199630513i"), new CNumber("-0.19077807856272289-0.5585597123976269i"), new CNumber("-0.5294253993237487-0.7455315910395643i"), new CNumber("0.2503474806442201")};
        expRowIndices = new int[]{0, 0, 1, 2};
        expColIndices = new int[]{1, 3, 1, 0};
        exp = new SparseCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, A.sub(B));

        // ------------------- Sub-case 3 -------------------
        aShape = new Shape(9, 5);
        aEntries = new double[]{0.6270128869218086, 0.16240110664574325, 0.45932672845870137, 0.7474735763620591, 0.538473104946311, 0.7052133339436125, 0.2592123105328553, 0.3787513850108264, 0.6772122005094601};
        aRowIndices = new int[]{1, 2, 3, 3, 4, 4, 5, 5, 8};
        aColIndices = new int[]{4, 0, 0, 2, 2, 4, 1, 4, 3};
        A = new SparseMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3, 5);
        bEntries = new CNumber[]{new CNumber("0.5868276372946869+0.2596163134750755i"), new CNumber("0.17707162901335272+0.6257920740208713i"), new CNumber("0.505831932260592+0.10185702848481692i")};
        bRowIndices = new int[]{0, 0, 2};
        bColIndices = new int[]{0, 4, 1};
        B = new SparseCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        SparseCMatrix finalB = B;
        assertThrows(Exception.class, ()->A.sub(finalB));
    }


    @Test
    void realSparseRealDenseSubTest() {
        double[][] bEntries;
        Matrix B;
        double[][] expEntries;
        Matrix exp;

        // ------------------- Sub-case 1 -------------------
        aShape = new Shape(5, 5);
        aEntries = new double[]{0.6121384636921453, 0.8740186531145545, 0.029381038939396142, 0.018338236585478507, 0.8225690346245645};
        aRowIndices = new int[]{1, 1, 2, 3, 3};
        aColIndices = new int[]{1, 2, 1, 1, 3};
        A = new SparseMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.01532, 0.57101, 0.9194, 0.62077, 0.16928},
                {0.36628, 0.44294, 0.77413, 0.27968, 0.14308},
                {0.11866, 0.19659, 0.20545, 0.95883, 0.19763},
                {0.89007, 0.78864, 0.66616, 0.88569, 0.21498},
                {0.08218, 0.4592, 0.39446, 0.76215, 0.09589}};
        B = new Matrix(bEntries);

        expEntries = new double[][]{
                {-0.01532, -0.57101, -0.9194, -0.62077, -0.16928},
                {-0.36628, 0.1691984636921453, 0.09988865311455453, -0.27968, -0.14308},
                {-0.11866, -0.16720896106060384, -0.20545, -0.95883, -0.19763},
                {-0.89007, -0.7703017634145215, -0.66616, -0.06312096537543543, -0.21498},
                {-0.08218, -0.4592, -0.39446, -0.76215, -0.09589}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.sub(B));

        // ------------------- Sub-case 2 -------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.6453030157220373, 0.046510654054616074, 0.26315624640775137};
        aRowIndices = new int[]{1, 1, 2};
        aColIndices = new int[]{0, 4, 2};
        A = new SparseMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.60345, 0.38265, 0.93371, 0.8198, 0.58958},
                {0.27596, 0.04954, 0.97929, 0.95783, 0.12248},
                {0.29138, 0.92012, 0.0453, 0.19584, 0.80848}};
        B = new Matrix(bEntries);

        expEntries = new double[][]{
                {-0.60345, -0.38265, -0.93371, -0.8198, -0.58958},
                {0.3693430157220373, -0.04954, -0.97929, -0.95783, -0.07596934594538393},
                {-0.29138, -0.92012, 0.21785624640775136, -0.19584, -0.80848}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.sub(B));

        // ------------------- Sub-case 3 -------------------
        aShape = new Shape(9, 5);
        aEntries = new double[]{0.5155800494513947, 0.546713180697451, 0.7643616937716915, 0.5419056469988012, 0.4939928805261391, 0.8767329019553655, 0.7469740304018805, 0.7002667865676956, 0.2988567558850965};
        aRowIndices = new int[]{0, 1, 1, 2, 4, 6, 8, 8, 8};
        aColIndices = new int[]{0, 1, 4, 1, 2, 1, 1, 2, 4};
        A = new SparseMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.61128, 0.19402, 0.09326, 0.9641, 0.31154},
                {0.33576, 0.2249, 0.27908, 0.38607, 0.80359},
                {0.80024, 0.14538, 0.48963, 0.01383, 0.61448}};
        B = new Matrix(bEntries);

        Matrix finalB = B;
        assertThrows(Exception.class, ()->A.sub(finalB));
    }


    @Test
    void realSparseComplexDenseSubTest() {
        CNumber[][] bEntries;
        CMatrix B;
        CNumber[][] expEntries;
        CMatrix exp;

        // ------------------- Sub-case 1 -------------------
        aShape = new Shape(5, 5);
        aEntries = new double[]{0.5179066536182447, 0.29699462002610133, 0.1384083495187035, 0.8642135522873594, 0.38365259331915025};
        aRowIndices = new int[]{2, 3, 3, 4, 4};
        aColIndices = new int[]{1, 0, 4, 0, 1};
        A = new SparseMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[][]{
                {new CNumber("0.75637+0.83187i"), new CNumber("0.31084+0.34292i"), new CNumber("0.25448+0.05582i"), new CNumber("0.2331+0.79232i"), new CNumber("0.89203+0.76204i")},
                {new CNumber("0.66323+0.60234i"), new CNumber("0.77748+0.54931i"), new CNumber("0.88589+0.76489i"), new CNumber("0.2303+0.0718i"), new CNumber("0.46611+0.1212i")},
                {new CNumber("0.29809+0.28872i"), new CNumber("0.3664+0.65027i"), new CNumber("0.81276+0.94666i"), new CNumber("0.83843+0.8782i"), new CNumber("0.3193+0.76019i")},
                {new CNumber("0.21547+0.89237i"), new CNumber("0.95833+0.90932i"), new CNumber("0.51026+0.15335i"), new CNumber("0.86221+0.93154i"), new CNumber("0.70569+0.87415i")},
                {new CNumber("0.12112+0.58718i"), new CNumber("0.55609+0.2181i"), new CNumber("0.61738+0.18952i"), new CNumber("0.51587+0.89177i"), new CNumber("0.76042+0.31481i")}};
        B = new CMatrix(bEntries);

        expEntries = new CNumber[][]{
                {new CNumber("-0.75637-0.83187i"), new CNumber("-0.31084-0.34292i"), new CNumber("-0.25448-0.05582i"), new CNumber("-0.2331-0.79232i"), new CNumber("-0.89203-0.76204i")},
                {new CNumber("-0.66323-0.60234i"), new CNumber("-0.77748-0.54931i"), new CNumber("-0.88589-0.76489i"), new CNumber("-0.2303-0.0718i"), new CNumber("-0.46611-0.1212i")},
                {new CNumber("-0.29809-0.28872i"), new CNumber("0.15150665361824472-0.65027i"), new CNumber("-0.81276-0.94666i"), new CNumber("-0.83843-0.8782i"), new CNumber("-0.3193-0.76019i")},
                {new CNumber("0.08152462002610134-0.89237i"), new CNumber("-0.95833-0.90932i"), new CNumber("-0.51026-0.15335i"), new CNumber("-0.86221-0.93154i"), new CNumber("-0.5672816504812965-0.87415i")},
                {new CNumber("0.7430935522873594-0.58718i"), new CNumber("-0.17243740668084973-0.2181i"), new CNumber("-0.61738-0.18952i"), new CNumber("-0.51587-0.89177i"), new CNumber("-0.76042-0.31481i")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.sub(B));

        // ------------------- Sub-case 2 -------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.11496605515101443, 0.781677500733315, 0.8839774286026595};
        aRowIndices = new int[]{0, 0, 1};
        aColIndices = new int[]{2, 3, 0};
        A = new SparseMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[][]{
                {new CNumber("0.52624+0.2508i"), new CNumber("0.83235+0.29734i"), new CNumber("0.92588+0.90113i"), new CNumber("0.38166+0.20651i"), new CNumber("0.78835+0.65337i")},
                {new CNumber("0.73068+0.81259i"), new CNumber("0.79852+0.93415i"), new CNumber("0.13249+0.03554i"), new CNumber("0.31218+0.74991i"), new CNumber("0.03226+0.56384i")},
                {new CNumber("0.53334+0.51873i"), new CNumber("0.45876+0.354i"), new CNumber("0.53354+0.6198i"), new CNumber("0.76748+0.68158i"), new CNumber("0.87113+0.69653i")}};
        B = new CMatrix(bEntries);

        expEntries = new CNumber[][]{
                {new CNumber("-0.52624-0.2508i"), new CNumber("-0.83235-0.29734i"), new CNumber("-0.8109139448489856-0.90113i"), new CNumber("0.400017500733315-0.20651i"), new CNumber("-0.78835-0.65337i")},
                {new CNumber("0.1532974286026595-0.81259i"), new CNumber("-0.79852-0.93415i"), new CNumber("-0.13249-0.03554i"), new CNumber("-0.31218-0.74991i"), new CNumber("-0.03226-0.56384i")},
                {new CNumber("-0.53334-0.51873i"), new CNumber("-0.45876-0.354i"), new CNumber("-0.53354-0.6198i"), new CNumber("-0.76748-0.68158i"), new CNumber("-0.87113-0.69653i")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.sub(B));

        // ------------------- Sub-case 3 -------------------
        aShape = new Shape(9, 5);
        aEntries = new double[]{0.12806056946267386, 0.3115401448576318, 0.22374771721866793, 0.17735659189845965, 0.3894332597306345, 0.6439604056373636, 0.843316756247472, 0.4266067850722429, 0.9094096898230055};
        aRowIndices = new int[]{0, 3, 3, 4, 4, 4, 5, 6, 7};
        aColIndices = new int[]{0, 1, 4, 0, 3, 4, 2, 2, 1};
        A = new SparseMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[][]{
                {new CNumber("0.96146+0.25417i"), new CNumber("0.38568+0.95976i"), new CNumber("0.35263+0.51317i"), new CNumber("0.16861+0.32575i"), new CNumber("0.45787+0.60226i")},
                {new CNumber("0.75246+0.26293i"), new CNumber("0.4652+0.18691i"), new CNumber("0.30874+0.838i"), new CNumber("0.85089+0.18647i"), new CNumber("0.55567+0.09379i")},
                {new CNumber("0.81861+0.37771i"), new CNumber("0.01698+0.29668i"), new CNumber("0.05255+0.14805i"), new CNumber("0.57575+0.25943i"), new CNumber("0.92083+0.02978i")}};
        B = new CMatrix(bEntries);

        CMatrix finalB = B;
        assertThrows(Exception.class, ()->A.sub(finalB));
    }
}
