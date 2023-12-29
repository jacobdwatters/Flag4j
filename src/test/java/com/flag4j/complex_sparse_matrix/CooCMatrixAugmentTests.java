package com.flag4j.complex_sparse_matrix;

import com.flag4j.*;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooCMatrixAugmentTests {

    @Test
    void realSparseAugmentTest() {
        Shape bShape;
        int[] bRowIndices;
        int[] bColIndices;
        CNumber[] bEntries;
        CooCMatrix b;

        Shape cShape;
        int[] cRowIndices;
        int[] cColIndices;
        double[] cEntries;
        CooMatrix c;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        CooCMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        bShape = new Shape(3, 2);
        bEntries = new CNumber[]{new CNumber("0.84785+0.65961i")};
        bRowIndices = new int[]{2};
        bColIndices = new int[]{0};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        cShape = new Shape(3, 4);
        cEntries = new double[]{0.54711, 0.0956};
        cRowIndices = new int[]{1, 2};
        cColIndices = new int[]{1, 2};
        c = new CooMatrix(cShape, cEntries, cRowIndices, cColIndices);

        expShape = new Shape(3, 6);
        expEntries = new CNumber[]{new CNumber("0.54711"), new CNumber("0.84785+0.65961i"), new CNumber("0.0956")};
        expRowIndices = new int[]{1, 2, 2};
        expColIndices = new int[]{3, 0, 4};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, b.augment(c));

        // ---------------------  Sub-case 2 ---------------------
        bShape = new Shape(2, 1);
        bEntries = new CNumber[]{};
        bRowIndices = new int[]{};
        bColIndices = new int[]{};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        cShape = new Shape(2, 5);
        cEntries = new double[]{0.11012, 0.05635};
        cRowIndices = new int[]{0, 1};
        cColIndices = new int[]{1, 4};
        c = new CooMatrix(cShape, cEntries, cRowIndices, cColIndices);

        expShape = new Shape(2, 6);
        expEntries = new CNumber[]{new CNumber("0.11012"), new CNumber("0.05635")};
        expRowIndices = new int[]{0, 1};
        expColIndices = new int[]{2, 5};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, b.augment(c));

        // ---------------------  Sub-case 3 ---------------------
        bShape = new Shape(5, 14);
        bEntries = new CNumber[]{new CNumber("0.20701+0.87362i"), new CNumber("0.99981+0.30156i"), new CNumber("0.59447+0.9851i"), new CNumber("0.59934+0.03263i"), new CNumber("0.8151+0.29466i"), new CNumber("0.99307+0.28989i"), new CNumber("0.53224+0.11927i"), new CNumber("0.94235+0.30785i"), new CNumber("0.40525+0.58865i"), new CNumber("0.97904+0.92044i"), new CNumber("0.21766+0.52526i"), new CNumber("0.94594+0.3945i"), new CNumber("0.64977+0.97097i"), new CNumber("0.70112+0.12694i")};
        bRowIndices = new int[]{0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4};
        bColIndices = new int[]{0, 1, 1, 4, 13, 1, 3, 9, 12, 6, 7, 4, 5, 6};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        cShape = new Shape(6, 14);
        cEntries = new double[]{0.2779, 0.48408, 0.54389, 0.09072, 0.90854, 0.18127, 0.57206, 0.80702, 0.42993, 0.84323, 0.96082, 0.38449, 0.42693, 0.55268, 0.03662, 0.21952, 0.36737};
        cRowIndices = new int[]{0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 5, 5};
        cColIndices = new int[]{9, 12, 8, 9, 10, 11, 1, 5, 8, 10, 0, 6, 7, 13, 13, 0, 9};
        c = new CooMatrix(cShape, cEntries, cRowIndices, cColIndices);

        CooCMatrix finalb = b;
        CooMatrix finalc = c;
        assertThrows(Exception.class, ()->finalb.augment(finalc));
    }


    @Test
    void complexSparseAugmentTest() {
        Shape bShape;
        int[] bRowIndices;
        int[] bColIndices;
        CNumber[] bEntries;
        CooCMatrix b;

        Shape dShape;
        int[] dRowIndices;
        int[] dColIndices;
        CNumber[] dEntries;
        CooCMatrix d;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        CooCMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        bShape = new Shape(3, 2);
        bEntries = new CNumber[]{new CNumber("0.31432+0.12231i")};
        bRowIndices = new int[]{0};
        bColIndices = new int[]{1};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        dShape = new Shape(3, 4);
        dEntries = new CNumber[]{new CNumber("0.7633+0.64041i"), new CNumber("0.31419+0.11711i")};
        dRowIndices = new int[]{0, 2};
        dColIndices = new int[]{2, 3};
        d = new CooCMatrix(dShape, dEntries, dRowIndices, dColIndices);

        expShape = new Shape(3, 6);
        expEntries = new CNumber[]{new CNumber("0.31432+0.12231i"), new CNumber("0.7633+0.64041i"), new CNumber("0.31419+0.11711i")};
        expRowIndices = new int[]{0, 0, 2};
        expColIndices = new int[]{1, 4, 5};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, b.augment(d));

        // ---------------------  Sub-case 2 ---------------------
        bShape = new Shape(2, 1);
        bEntries = new CNumber[]{};
        bRowIndices = new int[]{};
        bColIndices = new int[]{};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        dShape = new Shape(2, 5);
        dEntries = new CNumber[]{new CNumber("0.64596+0.34447i"), new CNumber("0.19735+0.00648i")};
        dRowIndices = new int[]{0, 0};
        dColIndices = new int[]{1, 3};
        d = new CooCMatrix(dShape, dEntries, dRowIndices, dColIndices);

        expShape = new Shape(2, 6);
        expEntries = new CNumber[]{new CNumber("0.64596+0.34447i"), new CNumber("0.19735+0.00648i")};
        expRowIndices = new int[]{0, 0};
        expColIndices = new int[]{2, 4};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, b.augment(d));

        // ---------------------  Sub-case 3 ---------------------
        bShape = new Shape(5, 14);
        bEntries = new CNumber[]{new CNumber("0.61976+0.88453i"), new CNumber("0.79514+0.80535i"), new CNumber("0.81806+0.71575i"), new CNumber("0.4667+0.1409i"), new CNumber("0.42131+0.5932i"), new CNumber("0.74726+0.137i"), new CNumber("0.33884+0.75794i"), new CNumber("0.8802+0.65175i"), new CNumber("0.81513+0.70436i"), new CNumber("0.86364+0.37206i"), new CNumber("0.54062+0.81757i"), new CNumber("0.66025+0.76792i"), new CNumber("0.88691+0.59128i"), new CNumber("0.82567+0.90133i")};
        bRowIndices = new int[]{0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4};
        bColIndices = new int[]{5, 8, 9, 5, 8, 2, 7, 9, 12, 13, 2, 5, 8, 11};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        dShape = new Shape(6, 14);
        dEntries = new CNumber[]{new CNumber("0.18121+0.09923i"), new CNumber("0.40046+0.45834i"), new CNumber("0.70669+0.49123i"), new CNumber("0.14628+0.52565i"), new CNumber("0.07834+0.83063i"), new CNumber("0.76763+0.26101i"), new CNumber("0.61068+0.5401i"), new CNumber("0.68096+0.10431i"), new CNumber("0.7411+0.29505i"), new CNumber("0.78902+0.40017i"), new CNumber("0.29019+0.90398i"), new CNumber("0.08451+0.94349i"), new CNumber("0.15869+0.17056i"), new CNumber("0.79893+0.59939i"), new CNumber("0.71202+0.00529i"), new CNumber("0.89399+0.9254i"), new CNumber("0.71421+0.37517i")};
        dRowIndices = new int[]{0, 1, 2, 2, 2, 2, 2, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5};
        dColIndices = new int[]{10, 3, 0, 2, 5, 7, 8, 0, 1, 3, 4, 10, 5, 7, 8, 12, 13};
        d = new CooCMatrix(dShape, dEntries, dRowIndices, dColIndices);

        CooCMatrix finalb = b;
        CooCMatrix finald = d;
        assertThrows(Exception.class, ()->finalb.augment(finald));
    }


    @Test
    void realDenseAugmentTest() {
        Shape bShape;
        int[] bRowIndices;
        int[] bColIndices;
        CNumber[] bEntries;
        CooCMatrix b;

        double[][] cEntries;
        Matrix c;

        CNumber[][] expEntries;
        CMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        bShape = new Shape(3, 2);
        bEntries = new CNumber[]{new CNumber("0.70332+0.55839i")};
        bRowIndices = new int[]{1};
        bColIndices = new int[]{0};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        cEntries = new double[][]{
                {0.61068, 0.60777, 0.55419, 0.12957},
                {0.13771, 0.66523, 0.76222, 0.06224},
                {0.81719, 0.95, 0.99729, 0.83164}};
        c = new Matrix(cEntries);

        expEntries = new CNumber[][]{
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.61068"), new CNumber("0.60777"), new CNumber("0.55419"), new CNumber("0.12957")},
                {new CNumber("0.70332+0.55839i"), new CNumber("0.0"), new CNumber("0.13771"), new CNumber("0.66523"), new CNumber("0.76222"), new CNumber("0.06224")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.81719"), new CNumber("0.95"), new CNumber("0.99729"), new CNumber("0.83164")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, b.augment(c));

        // ---------------------  Sub-case 2 ---------------------
        bShape = new Shape(2, 1);
        bEntries = new CNumber[]{};
        bRowIndices = new int[]{};
        bColIndices = new int[]{};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        cEntries = new double[][]{
                {0.7245, 0.90867, 0.30543, 0.2232, 0.30479},
                {0.88219, 0.56602, 0.53205, 0.47832, 0.82836}};
        c = new Matrix(cEntries);

        expEntries = new CNumber[][]{
                {new CNumber("0.0"), new CNumber("0.7245"), new CNumber("0.90867"), new CNumber("0.30543"), new CNumber("0.2232"), new CNumber("0.30479")},
                {new CNumber("0.0"), new CNumber("0.88219"), new CNumber("0.56602"), new CNumber("0.53205"), new CNumber("0.47832"), new CNumber("0.82836")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, b.augment(c));

        // ---------------------  Sub-case 3 ---------------------
        bShape = new Shape(5, 14);
        bEntries = new CNumber[]{new CNumber("0.4611+0.98524i"), new CNumber("0.49261+0.16759i"), new CNumber("0.08117+0.50377i"), new CNumber("0.68817+0.11529i"), new CNumber("0.94653+0.08864i"), new CNumber("0.84725+0.61267i"), new CNumber("0.3057+0.18656i"), new CNumber("0.37292+0.24222i"), new CNumber("0.55928+0.08697i"), new CNumber("0.94167+0.1131i"), new CNumber("0.3654+0.90583i"), new CNumber("0.42072+0.23239i"), new CNumber("0.58281+0.21006i"), new CNumber("0.41566+0.6451i")};
        bRowIndices = new int[]{0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4};
        bColIndices = new int[]{5, 11, 8, 9, 1, 5, 7, 9, 5, 13, 0, 3, 4, 11};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        cEntries = new double[][]{
                {0.41175, 0.39957, 0.78298, 0.9453, 0.27168, 0.55028, 0.61166, 0.37744, 0.0839, 0.19246, 0.36171, 0.29614, 0.15624, 0.30183},
                {0.13372, 0.56701, 0.96003, 0.24742, 0.88218, 0.31421, 0.70881, 0.44273, 0.89411, 0.54812, 0.75252, 0.18215, 0.84299, 0.35466},
                {0.81377, 0.9046, 0.39991, 0.53753, 0.57927, 0.54059, 0.88129, 0.42077, 0.70458, 0.45264, 0.10284, 0.4169, 0.20322, 0.10635},
                {0.16339, 0.99854, 0.25237, 0.49311, 0.15197, 0.31013, 0.27184, 0.43532, 0.59896, 0.03028, 0.91806, 0.17735, 0.50379, 0.57811},
                {0.72902, 0.15129, 0.25434, 0.64536, 0.73076, 0.03561, 0.86844, 0.40655, 0.79753, 0.6789, 0.31988, 0.47152, 0.14378, 0.06038},
                {0.72641, 0.56986, 0.1297, 0.73257, 0.66233, 0.02298, 0.35783, 0.35666, 0.52759, 0.68261, 0.46329, 0.25232, 0.02668, 0.87163}};
        c = new Matrix(cEntries);

        CooCMatrix finalb = b;
        Matrix finalc = c;
        assertThrows(Exception.class, ()->finalb.augment(finalc));
    }


    @Test
    void complexDenseAugmentTest() {
        Shape bShape;
        int[] bRowIndices;
        int[] bColIndices;
        CNumber[] bEntries;
        CooCMatrix b;

        CNumber[][] fEntries;
        CMatrix f;

        CNumber[][] expEntries;
        CMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        bShape = new Shape(3, 2);
        bEntries = new CNumber[]{new CNumber("0.25154+0.92877i")};
        bRowIndices = new int[]{2};
        bColIndices = new int[]{1};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        fEntries = new CNumber[][]{
                {new CNumber("0.92981+0.23361i"), new CNumber("0.93287+0.43796i"), new CNumber("0.59093+0.25931i"), new CNumber("0.21694+0.82226i")},
                {new CNumber("0.79908+0.71528i"), new CNumber("0.91744+0.87534i"), new CNumber("0.45513+0.67579i"), new CNumber("0.66358+0.91274i")},
                {new CNumber("0.67231+0.88016i"), new CNumber("0.00145+0.62177i"), new CNumber("0.12264+0.0375i"), new CNumber("0.55567+0.0237i")}};
        f = new CMatrix(fEntries);

        expEntries = new CNumber[][]{
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.92981+0.23361i"), new CNumber("0.93287+0.43796i"), new CNumber("0.59093+0.25931i"), new CNumber("0.21694+0.82226i")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.79908+0.71528i"), new CNumber("0.91744+0.87534i"), new CNumber("0.45513+0.67579i"), new CNumber("0.66358+0.91274i")},
                {new CNumber("0.0"), new CNumber("0.25154+0.92877i"), new CNumber("0.67231+0.88016i"), new CNumber("0.00145+0.62177i"), new CNumber("0.12264+0.0375i"), new CNumber("0.55567+0.0237i")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, b.augment(f));

        // ---------------------  Sub-case 2 ---------------------
        bShape = new Shape(2, 1);
        bEntries = new CNumber[]{};
        bRowIndices = new int[]{};
        bColIndices = new int[]{};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        fEntries = new CNumber[][]{
                {new CNumber("0.88652+0.93435i"), new CNumber("0.32431+0.93111i"), new CNumber("0.48725+0.25879i"), new CNumber("0.34835+0.79514i"), new CNumber("0.60759+0.40738i")},
                {new CNumber("0.67538+0.18465i"), new CNumber("0.15309+0.24463i"), new CNumber("0.66734+0.5884i"), new CNumber("0.86899+0.939i"), new CNumber("0.89835+0.36132i")}};
        f = new CMatrix(fEntries);

        expEntries = new CNumber[][]{
                {new CNumber("0.0"), new CNumber("0.88652+0.93435i"), new CNumber("0.32431+0.93111i"), new CNumber("0.48725+0.25879i"), new CNumber("0.34835+0.79514i"), new CNumber("0.60759+0.40738i")},
                {new CNumber("0.0"), new CNumber("0.67538+0.18465i"), new CNumber("0.15309+0.24463i"), new CNumber("0.66734+0.5884i"), new CNumber("0.86899+0.939i"), new CNumber("0.89835+0.36132i")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, b.augment(f));

        // ---------------------  Sub-case 3 ---------------------
        bShape = new Shape(5, 14);
        bEntries = new CNumber[]{new CNumber("0.39508+0.48337i"), new CNumber("0.59284+0.95713i"), new CNumber("0.13141+0.88985i"), new CNumber("0.0529+0.24701i"), new CNumber("0.15004+0.6474i"), new CNumber("0.72546+0.9462i"), new CNumber("0.95843+0.16119i"), new CNumber("0.88454+0.93482i"), new CNumber("0.01919+0.33417i"), new CNumber("0.58352+0.93151i"), new CNumber("0.29506+0.2507i"), new CNumber("0.00409+0.22329i"), new CNumber("0.64064+0.42262i"), new CNumber("0.13876+0.72422i")};
        bRowIndices = new int[]{0, 0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4};
        bColIndices = new int[]{0, 4, 5, 0, 4, 6, 7, 7, 2, 8, 12, 1, 2, 9};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        fEntries = new CNumber[][]{
                {new CNumber("0.33631+0.85681i"), new CNumber("0.75917+0.05944i"), new CNumber("0.68038+0.98755i"), new CNumber("0.22031+0.53613i"), new CNumber("0.79342+0.31851i"), new CNumber("0.83525+0.80468i"), new CNumber("0.49668+0.87247i"), new CNumber("0.95718+0.5632i"), new CNumber("0.38981+0.49794i"), new CNumber("0.55703+0.78395i"), new CNumber("0.47575+0.88674i"), new CNumber("0.08335+0.88026i"), new CNumber("0.7018+0.10366i"), new CNumber("0.38898+0.01308i")},
                {new CNumber("0.92157+0.17295i"), new CNumber("0.22122+0.40573i"), new CNumber("0.91258+0.12746i"), new CNumber("0.33346+0.44264i"), new CNumber("0.85873+0.66793i"), new CNumber("0.67968+0.98715i"), new CNumber("0.28553+0.99211i"), new CNumber("0.61244+0.48105i"), new CNumber("0.89077+0.99372i"), new CNumber("0.20145+0.76687i"), new CNumber("0.46331+0.85529i"), new CNumber("0.43793+0.26793i"), new CNumber("0.87893+0.09417i"), new CNumber("0.45766+0.82012i")},
                {new CNumber("0.14704+0.21164i"), new CNumber("0.28336+0.64859i"), new CNumber("0.68308+0.34816i"), new CNumber("0.58468+0.96637i"), new CNumber("0.18303+0.86332i"), new CNumber("0.87873+0.09218i"), new CNumber("0.01388+0.43583i"), new CNumber("0.3806+0.04051i"), new CNumber("0.92944+0.85833i"), new CNumber("0.5818+0.25126i"), new CNumber("0.99908+0.1199i"), new CNumber("0.94264+0.33433i"), new CNumber("0.46006+0.11749i"), new CNumber("0.46772+0.7015i")},
                {new CNumber("0.37042+0.15538i"), new CNumber("0.32728+0.70955i"), new CNumber("0.89525+0.30299i"), new CNumber("0.50943+0.9311i"), new CNumber("0.35604+0.99776i"), new CNumber("0.08328+0.70028i"), new CNumber("0.86287+0.583i"), new CNumber("0.77376+0.59036i"), new CNumber("0.802+0.16701i"), new CNumber("0.24959+0.63032i"), new CNumber("0.27946+0.1442i"), new CNumber("0.9216+0.67413i"), new CNumber("0.5035+0.86376i"), new CNumber("0.23092+0.62888i")},
                {new CNumber("0.62384+0.53906i"), new CNumber("0.13778+0.39585i"), new CNumber("0.7412+0.86774i"), new CNumber("0.98219+0.29473i"), new CNumber("0.35404+0.85705i"), new CNumber("0.99046+0.99094i"), new CNumber("0.22903+0.08892i"), new CNumber("0.56313+0.30066i"), new CNumber("0.154+0.66633i"), new CNumber("0.62317+0.53893i"), new CNumber("0.80426+0.41639i"), new CNumber("0.45131+0.15304i"), new CNumber("0.90697+0.88845i"), new CNumber("0.0908+0.17038i")},
                {new CNumber("0.1367+0.23041i"), new CNumber("0.37317+0.37023i"), new CNumber("0.59457+0.67128i"), new CNumber("0.82021+0.65897i"), new CNumber("0.79675+0.98153i"), new CNumber("0.22047+0.1856i"), new CNumber("0.60456+0.95211i"), new CNumber("0.8157+0.20811i"), new CNumber("0.82446+0.64386i"), new CNumber("0.95599+0.62204i"), new CNumber("0.49704+0.39421i"), new CNumber("0.45722+0.8548i"), new CNumber("0.15736+0.22322i"), new CNumber("0.62343+0.55472i")}};
        f = new CMatrix(fEntries);

        CooCMatrix finalb = b;
        CMatrix finalf = f;
        assertThrows(Exception.class, ()->finalb.augment(finalf));
    }
}
