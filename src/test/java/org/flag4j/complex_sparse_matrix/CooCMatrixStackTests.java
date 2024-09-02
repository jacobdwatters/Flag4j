package org.flag4j.complex_sparse_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.sparse.CooCMatrixOld;
import org.flag4j.arrays_old.sparse.CooMatrixOld;
import org.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooCMatrixStackTests {

    @Test
    void realSparseStackTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        CooCMatrixOld a;

        Shape bShape;
        int[] bRowIndices;
        int[] bColIndices;
        double[] bEntries;
        CooMatrixOld b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        CooCMatrixOld exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(2, 3);
        aEntries = new CNumber[]{new CNumber("0.39505+0.21766i")};
        aRowIndices = new int[]{1};
        aColIndices = new int[]{1};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(4, 3);
        bEntries = new double[]{0.92973, 0.47392};
        bRowIndices = new int[]{2, 3};
        bColIndices = new int[]{1, 2};
        b = new CooMatrixOld(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(6, 3);
        expEntries = new CNumber[]{new CNumber("0.39505+0.21766i"), new CNumber("0.92973"), new CNumber("0.47392")};
        expRowIndices = new int[]{1, 4, 5};
        expColIndices = new int[]{1, 1, 2};
        exp = new CooCMatrixOld(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.stack(b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(1, 2);
        aEntries = new CNumber[]{};
        aRowIndices = new int[]{};
        aColIndices = new int[]{};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 2);
        bEntries = new double[]{0.25966, 0.57659};
        bRowIndices = new int[]{0, 1};
        bColIndices = new int[]{0, 1};
        b = new CooMatrixOld(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(6, 2);
        expEntries = new CNumber[]{new CNumber("0.25966"), new CNumber("0.57659")};
        expRowIndices = new int[]{1, 2};
        expColIndices = new int[]{0, 1};
        exp = new CooCMatrixOld(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.stack(b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(14, 5);
        aEntries = new CNumber[]{new CNumber("0.30936+0.11429i"), new CNumber("0.82155+0.23318i"), new CNumber("0.80561+0.24765i"), new CNumber("0.41344+0.05571i"), new CNumber("0.06089+0.29535i"), new CNumber("0.75737+0.21016i"), new CNumber("0.8582+0.6529i"), new CNumber("0.40702+0.95626i"), new CNumber("0.21659+0.81609i"), new CNumber("0.78506+0.73484i"), new CNumber("0.36489+0.12724i"), new CNumber("0.33911+0.6715i"), new CNumber("0.87291+0.32998i"), new CNumber("0.7809+0.51503i")};
        aRowIndices = new int[]{0, 1, 1, 2, 3, 4, 4, 5, 5, 8, 10, 11, 11, 13};
        aColIndices = new int[]{4, 1, 2, 0, 4, 1, 4, 0, 1, 1, 1, 1, 2, 4};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(14, 6);
        bEntries = new double[]{0.00364, 0.07403, 0.5977, 0.19641, 0.31155, 0.83437, 0.49562, 0.75274, 0.58422, 0.00951, 0.80351, 0.47327, 0.4818, 0.71602, 0.14986, 0.34991, 0.84375};
        bRowIndices = new int[]{1, 3, 3, 5, 6, 7, 7, 8, 8, 8, 9, 10, 10, 11, 11, 12, 12};
        bColIndices = new int[]{3, 0, 4, 2, 3, 3, 4, 1, 3, 5, 4, 1, 3, 2, 5, 3, 4};
        b = new CooMatrixOld(bShape, bEntries, bRowIndices, bColIndices);

        CooCMatrixOld finala = a;
        CooMatrixOld finalb = b;
        assertThrows(Exception.class, ()->finala.stack(finalb));
    }


    @Test
    void complexSparseStackTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        CooCMatrixOld a;

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
        aEntries = new CNumber[]{new CNumber("0.25394+0.43087i")};
        aRowIndices = new int[]{0};
        aColIndices = new int[]{0};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(4, 3);
        bEntries = new CNumber[]{new CNumber("0.64208+0.02231i"), new CNumber("0.88585+0.80165i")};
        bRowIndices = new int[]{0, 2};
        bColIndices = new int[]{0, 0};
        b = new CooCMatrixOld(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(6, 3);
        expEntries = new CNumber[]{new CNumber("0.25394+0.43087i"), new CNumber("0.64208+0.02231i"), new CNumber("0.88585+0.80165i")};
        expRowIndices = new int[]{0, 2, 4};
        expColIndices = new int[]{0, 0, 0};
        exp = new CooCMatrixOld(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.stack(b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(1, 2);
        aEntries = new CNumber[]{};
        aRowIndices = new int[]{};
        aColIndices = new int[]{};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 2);
        bEntries = new CNumber[]{new CNumber("0.49415+0.73554i"), new CNumber("0.50848+0.69047i")};
        bRowIndices = new int[]{3, 3};
        bColIndices = new int[]{0, 1};
        b = new CooCMatrixOld(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(6, 2);
        expEntries = new CNumber[]{new CNumber("0.49415+0.73554i"), new CNumber("0.50848+0.69047i")};
        expRowIndices = new int[]{4, 4};
        expColIndices = new int[]{0, 1};
        exp = new CooCMatrixOld(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.stack(b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(14, 5);
        aEntries = new CNumber[]{new CNumber("0.28989+0.09294i"), new CNumber("0.14286+0.13982i"), new CNumber("0.08978+0.69954i"), new CNumber("0.25905+0.81205i"), new CNumber("0.74641+0.0583i"), new CNumber("0.59938+0.46717i"), new CNumber("0.26599+0.76388i"), new CNumber("0.27019+0.47096i"), new CNumber("0.91115+0.20502i"), new CNumber("0.01355+0.83376i"), new CNumber("0.26981+0.89026i"), new CNumber("0.99982+0.06107i"), new CNumber("0.89229+0.62325i"), new CNumber("0.44728+0.68547i")};
        aRowIndices = new int[]{0, 1, 1, 2, 2, 3, 3, 5, 6, 6, 7, 10, 10, 13};
        aColIndices = new int[]{2, 0, 1, 0, 2, 0, 1, 0, 0, 4, 4, 2, 3, 3};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(14, 6);
        bEntries = new CNumber[]{new CNumber("0.6588+0.75876i"), new CNumber("0.60697+0.64487i"), new CNumber("0.76965+0.38201i"), new CNumber("0.98363+0.84591i"), new CNumber("0.77316+0.25268i"), new CNumber("0.83365+0.37816i"), new CNumber("0.07067+0.88382i"), new CNumber("0.6195+0.37165i"), new CNumber("0.86337+0.12018i"), new CNumber("0.77066+0.0812i"), new CNumber("0.28691+0.60752i"), new CNumber("0.99723+0.46211i"), new CNumber("0.44258+0.43083i"), new CNumber("0.93875+0.27681i"), new CNumber("0.64751+0.18722i"), new CNumber("0.04095+0.24351i"), new CNumber("0.23496+0.90917i")};
        bRowIndices = new int[]{0, 0, 1, 4, 5, 5, 6, 7, 7, 8, 8, 8, 10, 10, 11, 12, 13};
        bColIndices = new int[]{0, 5, 0, 2, 0, 3, 4, 1, 3, 1, 4, 5, 0, 5, 1, 0, 2};
        b = new CooCMatrixOld(bShape, bEntries, bRowIndices, bColIndices);

        CooCMatrixOld finala = a;
        CooCMatrixOld finalb = b;
        assertThrows(Exception.class, ()->finala.stack(finalb));
    }


    @Test
    void realDenseStackTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        CooCMatrixOld a;

        double[][] bEntries;
        MatrixOld b;

        CNumber[][] expEntries;
        CMatrixOld exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(2, 3);
        aEntries = new CNumber[]{new CNumber("0.36404+0.33988i")};
        aRowIndices = new int[]{0};
        aColIndices = new int[]{0};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.67996, 0.53919, 0.78631},
                {0.1186, 0.37476, 0.16975},
                {0.76593, 0.75605, 0.36715},
                {0.94251, 0.73827, 0.94622}};
        b = new MatrixOld(bEntries);

        expEntries = new CNumber[][]{
                {new CNumber("0.36404+0.33988i"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.67996"), new CNumber("0.53919"), new CNumber("0.78631")},
                {new CNumber("0.1186"), new CNumber("0.37476"), new CNumber("0.16975")},
                {new CNumber("0.76593"), new CNumber("0.75605"), new CNumber("0.36715")},
                {new CNumber("0.94251"), new CNumber("0.73827"), new CNumber("0.94622")}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, a.stack(b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(1, 2);
        aEntries = new CNumber[]{};
        aRowIndices = new int[]{};
        aColIndices = new int[]{};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.25619, 0.00526},
                {0.89488, 0.46937},
                {0.93038, 0.71823},
                {0.58386, 0.41477},
                {0.63463, 0.94818}};
        b = new MatrixOld(bEntries);

        expEntries = new CNumber[][]{
                {new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.25619"), new CNumber("0.00526")},
                {new CNumber("0.89488"), new CNumber("0.46937")},
                {new CNumber("0.93038"), new CNumber("0.71823")},
                {new CNumber("0.58386"), new CNumber("0.41477")},
                {new CNumber("0.63463"), new CNumber("0.94818")}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, a.stack(b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(14, 5);
        aEntries = new CNumber[]{new CNumber("0.57748+0.47936i"), new CNumber("0.03435+0.10023i"), new CNumber("0.03198+0.69263i"), new CNumber("0.55742+0.93426i"), new CNumber("0.8556+0.20584i"), new CNumber("0.89845+0.99301i"), new CNumber("0.23372+0.80954i"), new CNumber("0.31915+0.93021i"), new CNumber("0.24882+0.42916i"), new CNumber("0.65295+0.12752i"), new CNumber("0.464+0.13734i"), new CNumber("0.88421+0.15809i"), new CNumber("0.51505+0.50713i"), new CNumber("0.87364+0.00411i")};
        aRowIndices = new int[]{0, 0, 0, 0, 1, 2, 4, 5, 6, 6, 9, 11, 11, 13};
        aColIndices = new int[]{0, 2, 3, 4, 0, 2, 3, 1, 3, 4, 3, 1, 2, 4};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.51986, 0.28629, 0.01035, 0.3259, 0.22019, 0.6306},
                {0.04073, 0.45149, 0.75424, 0.02617, 0.24855, 0.90368},
                {0.56287, 0.60915, 0.91277, 0.97026, 0.80892, 0.59717},
                {0.626, 0.59651, 0.88941, 0.05376, 0.23271, 0.13412},
                {0.46618, 0.09807, 0.50688, 0.88261, 0.14002, 0.80768},
                {0.95871, 0.01269, 0.70297, 0.69331, 0.75129, 0.98411},
                {0.99446, 0.07405, 0.28635, 0.30787, 0.55584, 0.99108},
                {0.8789, 0.82995, 0.18031, 0.8294, 0.68702, 0.92605},
                {0.6616, 0.99323, 0.43111, 0.19132, 0.61887, 0.05012},
                {0.23313, 0.57339, 0.09415, 0.48627, 0.08219, 0.30282},
                {0.57839, 0.43884, 0.8147, 0.83715, 0.47299, 0.99837},
                {0.61932, 0.04938, 0.47473, 0.63277, 0.60448, 0.33821},
                {0.38477, 0.28532, 0.73475, 0.44512, 0.14746, 0.99019},
                {0.34598, 0.91827, 0.63773, 0.76461, 0.66024, 0.31418}};
        b = new MatrixOld(bEntries);

        CooCMatrixOld finala = a;
        MatrixOld finalb = b;
        assertThrows(Exception.class, ()->finala.stack(finalb));
    }


    @Test
    void complexDenseStackTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        CooCMatrixOld a;

        CNumber[][] bEntries;
        CMatrixOld b;

        CNumber[][] expEntries;
        CMatrixOld exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(2, 3);
        aEntries = new CNumber[]{new CNumber("0.07899+0.54147i")};
        aRowIndices = new int[]{1};
        aColIndices = new int[]{0};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[][]{
                {new CNumber("0.67137+0.84205i"), new CNumber("0.28936+0.68917i"), new CNumber("0.11644+0.10053i")},
                {new CNumber("0.63929+0.17997i"), new CNumber("0.9858+0.31447i"), new CNumber("0.86033+0.26543i")},
                {new CNumber("0.03155+0.57001i"), new CNumber("0.70789+0.81428i"), new CNumber("0.22397+0.82456i")},
                {new CNumber("0.23551+0.25312i"), new CNumber("0.96902+0.93182i"), new CNumber("0.33595+0.11643i")}};
        b = new CMatrixOld(bEntries);

        expEntries = new CNumber[][]{
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.07899+0.54147i"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.67137+0.84205i"), new CNumber("0.28936+0.68917i"), new CNumber("0.11644+0.10053i")},
                {new CNumber("0.63929+0.17997i"), new CNumber("0.9858+0.31447i"), new CNumber("0.86033+0.26543i")},
                {new CNumber("0.03155+0.57001i"), new CNumber("0.70789+0.81428i"), new CNumber("0.22397+0.82456i")},
                {new CNumber("0.23551+0.25312i"), new CNumber("0.96902+0.93182i"), new CNumber("0.33595+0.11643i")}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, a.stack(b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(1, 2);
        aEntries = new CNumber[]{};
        aRowIndices = new int[]{};
        aColIndices = new int[]{};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[][]{
                {new CNumber("0.0688+0.87647i"), new CNumber("0.76826+0.22192i")},
                {new CNumber("0.07651+0.32947i"), new CNumber("0.55627+0.81767i")},
                {new CNumber("0.57404+0.24268i"), new CNumber("0.68701+0.69182i")},
                {new CNumber("0.75695+0.82641i"), new CNumber("0.91012+0.62214i")},
                {new CNumber("0.04513+0.61241i"), new CNumber("0.7108+0.09859i")}};
        b = new CMatrixOld(bEntries);

        expEntries = new CNumber[][]{
                {new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0688+0.87647i"), new CNumber("0.76826+0.22192i")},
                {new CNumber("0.07651+0.32947i"), new CNumber("0.55627+0.81767i")},
                {new CNumber("0.57404+0.24268i"), new CNumber("0.68701+0.69182i")},
                {new CNumber("0.75695+0.82641i"), new CNumber("0.91012+0.62214i")},
                {new CNumber("0.04513+0.61241i"), new CNumber("0.7108+0.09859i")}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, a.stack(b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(14, 5);
        aEntries = new CNumber[]{new CNumber("0.5671+0.53289i"), new CNumber("0.02133+0.78016i"), new CNumber("0.04768+0.89941i"), new CNumber("0.38962+0.5808i"), new CNumber("0.94924+0.67534i"), new CNumber("0.60143+0.75209i"), new CNumber("0.52088+0.60107i"), new CNumber("0.71471+0.61016i"), new CNumber("0.46824+0.57474i"), new CNumber("0.54599+0.99632i"), new CNumber("0.4496+0.84622i"), new CNumber("0.79449+0.77236i"), new CNumber("0.30146+0.45992i"), new CNumber("0.92227+0.40969i")};
        aRowIndices = new int[]{1, 1, 4, 4, 4, 4, 5, 5, 7, 7, 9, 11, 11, 12};
        aColIndices = new int[]{2, 3, 0, 1, 2, 4, 0, 2, 1, 4, 1, 1, 3, 3};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[][]{
                {new CNumber("0.77226+0.04976i"), new CNumber("0.08596+0.07924i"), new CNumber("0.09049+0.89063i"), new CNumber("0.60278+0.03141i"), new CNumber("0.51697+0.33241i"), new CNumber("0.72609+0.20861i")},
                {new CNumber("0.04882+0.19484i"), new CNumber("0.67273+0.47138i"), new CNumber("0.53016+0.05879i"), new CNumber("0.93508+0.28659i"), new CNumber("0.14167+0.78014i"), new CNumber("0.61638+0.262i")},
                {new CNumber("0.92285+0.12887i"), new CNumber("0.25818+0.65645i"), new CNumber("0.76882+0.29284i"), new CNumber("0.69965+0.80224i"), new CNumber("0.36369+0.78071i"), new CNumber("0.25069+0.83814i")},
                {new CNumber("0.10094+0.90308i"), new CNumber("0.35303+0.80533i"), new CNumber("0.04851+0.97201i"), new CNumber("0.81801+0.58789i"), new CNumber("0.34461+0.90112i"), new CNumber("0.32834+0.77718i")},
                {new CNumber("0.02177+0.45352i"), new CNumber("0.35679+0.66002i"), new CNumber("0.12725+0.83309i"), new CNumber("0.63381+0.02039i"), new CNumber("0.08612+0.20982i"), new CNumber("0.87178+0.67946i")},
                {new CNumber("0.01503+0.83737i"), new CNumber("0.65616+0.3176i"), new CNumber("0.64778+0.3044i"), new CNumber("0.00712+0.72631i"), new CNumber("0.24215+0.34366i"), new CNumber("0.88801+0.80612i")},
                {new CNumber("0.0509+0.41602i"), new CNumber("0.91904+0.69504i"), new CNumber("0.69234+0.8721i"), new CNumber("0.33789+0.28828i"), new CNumber("0.03886+0.40461i"), new CNumber("0.08576+0.42559i")},
                {new CNumber("0.36452+0.72538i"), new CNumber("0.18068+0.59048i"), new CNumber("0.21846+0.12914i"), new CNumber("0.15163+0.46358i"), new CNumber("0.46989+0.11226i"), new CNumber("0.85292+0.60804i")},
                {new CNumber("0.58723+0.565i"), new CNumber("0.60102+0.98269i"), new CNumber("0.72071+0.54211i"), new CNumber("0.67589+0.02896i"), new CNumber("0.73746+0.69418i"), new CNumber("0.36661+0.79122i")},
                {new CNumber("0.84557+0.77629i"), new CNumber("0.12704+0.73943i"), new CNumber("0.72172+0.2848i"), new CNumber("0.00487+0.14589i"), new CNumber("0.57025+0.23208i"), new CNumber("0.48652+0.7753i")},
                {new CNumber("0.63414+0.29602i"), new CNumber("0.77984+0.3253i"), new CNumber("0.51624+0.15347i"), new CNumber("0.81352+0.94694i"), new CNumber("0.3438+0.11788i"), new CNumber("0.98461+0.81196i")},
                {new CNumber("0.5352+0.07752i"), new CNumber("0.32892+0.35908i"), new CNumber("0.43592+0.20698i"), new CNumber("0.78715+0.83124i"), new CNumber("0.76193+0.85584i"), new CNumber("0.63135+0.59955i")},
                {new CNumber("0.47471+0.18006i"), new CNumber("0.74822+0.37998i"), new CNumber("0.08842+0.42187i"), new CNumber("0.27885+0.63167i"), new CNumber("0.49626+0.67038i"), new CNumber("0.72836+0.48923i")},
                {new CNumber("0.85682+0.02004i"), new CNumber("0.984+0.06694i"), new CNumber("0.31869+0.16227i"), new CNumber("0.58742+0.18035i"), new CNumber("0.25985+0.30886i"), new CNumber("0.37326+0.45303i")}};
        b = new CMatrixOld(bEntries);

        CooCMatrixOld finala = a;
        CMatrixOld finalb = b;
        assertThrows(Exception.class, ()->finala.stack(finalb));
    }
}
