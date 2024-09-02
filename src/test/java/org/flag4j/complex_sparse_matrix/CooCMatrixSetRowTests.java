package org.flag4j.complex_sparse_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.sparse.CooCMatrixOld;
import org.flag4j.arrays_old.sparse.CooCVectorOld;
import org.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooCMatrixSetRowTests {

    @Test
    void setRowSparseTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        CooCMatrixOld a;

        Shape bShape;
        int[] bindices;
        CNumber[] bEntries;
        CooCVectorOld b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        CooCMatrixOld exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.25598+0.65677i"), new CNumber("0.13544+0.86431i"), new CNumber("0.42906+0.88227i")};
        aRowIndices = new int[]{0, 1, 4};
        aColIndices = new int[]{0, 1, 2};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3);
        bEntries = new CNumber[]{new CNumber("0.18568+0.10002i")};
        bindices = new int[]{2};
        b = new CooCVectorOld(bShape.get(0), bEntries, bindices);

        expShape = new Shape(5, 3);
        expEntries = new CNumber[]{new CNumber("0.18568+0.10002i"), new CNumber("0.13544+0.86431i"), new CNumber("0.42906+0.88227i")};
        expRowIndices = new int[]{0, 1, 4};
        expColIndices = new int[]{2, 1, 2};
        exp = new CooCMatrixOld(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setRow(b, 0));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new CNumber[]{new CNumber("0.25343+0.33083i"), new CNumber("0.89061+0.1353i"), new CNumber("0.83798+0.5638i"), new CNumber("0.22808+0.99318i"), new CNumber("0.51957+0.88361i")};
        aRowIndices = new int[]{0, 3, 4, 6, 10};
        aColIndices = new int[]{14, 6, 14, 19, 11};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(23);
        bEntries = new CNumber[]{new CNumber("0.09868+0.82283i"), new CNumber("0.5747+0.68314i"), new CNumber("0.97588+0.37176i"), new CNumber("0.01652+0.29332i")};
        bindices = new int[]{2, 8, 16, 21};
        b = new CooCVectorOld(bShape.get(0), bEntries, bindices);

        expShape = new Shape(11, 23);
        expEntries = new CNumber[]{new CNumber("0.25343+0.33083i"), new CNumber("0.09868+0.82283i"), new CNumber("0.5747+0.68314i"), new CNumber("0.97588+0.37176i"), new CNumber("0.01652+0.29332i"), new CNumber("0.89061+0.1353i"), new CNumber("0.83798+0.5638i"), new CNumber("0.22808+0.99318i"), new CNumber("0.51957+0.88361i")};
        expRowIndices = new int[]{0, 1, 1, 1, 1, 3, 4, 6, 10};
        expColIndices = new int[]{14, 2, 8, 16, 21, 6, 14, 19, 11};
        exp = new CooCMatrixOld(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setRow(b, 1));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(5, 1000);
        aEntries = new CNumber[]{new CNumber("0.63307+0.18148i"), new CNumber("0.89993+0.94657i"), new CNumber("0.01684+0.17543i"), new CNumber("0.29048+0.35145i"), new CNumber("0.03111+0.05317i"), new CNumber("0.3826+0.46659i"), new CNumber("0.90015+0.78214i"), new CNumber("0.94882+0.99858i"), new CNumber("0.54192+0.60323i")};
        aRowIndices = new int[]{1, 2, 2, 2, 3, 3, 4, 4, 4};
        aColIndices = new int[]{562, 259, 319, 624, 386, 817, 367, 532, 693};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(1000);
        bEntries = new CNumber[]{new CNumber("0.83231+0.0157i"), new CNumber("0.94923+0.26837i"), new CNumber("0.15318+0.79913i"), new CNumber("0.8847+0.9145i"), new CNumber("0.05361+0.50878i")};
        bindices = new int[]{111, 401, 626, 664, 884};
        b = new CooCVectorOld(bShape.get(0), bEntries, bindices);

        expShape = new Shape(5, 1000);
        expEntries = new CNumber[]{new CNumber("0.63307+0.18148i"), new CNumber("0.89993+0.94657i"), new CNumber("0.01684+0.17543i"), new CNumber("0.29048+0.35145i"), new CNumber("0.03111+0.05317i"), new CNumber("0.3826+0.46659i"), new CNumber("0.83231+0.0157i"), new CNumber("0.94923+0.26837i"), new CNumber("0.15318+0.79913i"), new CNumber("0.8847+0.9145i"), new CNumber("0.05361+0.50878i")};
        expRowIndices = new int[]{1, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4};
        expColIndices = new int[]{562, 259, 319, 624, 386, 817, 111, 401, 626, 664, 884};
        exp = new CooCMatrixOld(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setRow(b, 4));

        // ---------------------  Sub-case 4 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.95261+0.18439i"), new CNumber("0.14661+0.17893i"), new CNumber("0.53326+0.75106i"), new CNumber("0.30531+0.81679i")};
        aRowIndices = new int[]{0, 0, 0, 2};
        aColIndices = new int[]{0, 1, 4, 4};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(4);
        bEntries = new CNumber[]{new CNumber("0.32913+0.94182i")};
        bindices = new int[]{1};
        b = new CooCVectorOld(bShape.get(0), bEntries, bindices);

        CooCMatrixOld final0a = a;
        CooCVectorOld final0b = b;
        assertThrows(Exception.class, ()->final0a.setRow(final0b, 2));

        // ---------------------  Sub-case 5 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.12926+0.41307i"), new CNumber("0.73917+0.34081i"), new CNumber("0.78819+0.9689i"), new CNumber("0.331+0.41141i")};
        aRowIndices = new int[]{0, 1, 2, 2};
        aColIndices = new int[]{3, 4, 1, 4};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(6);
        bEntries = new CNumber[]{new CNumber("0.10412+0.70557i"), new CNumber("0.193+0.2337i")};
        bindices = new int[]{1, 5};
        b = new CooCVectorOld(bShape.get(0), bEntries, bindices);

        CooCMatrixOld final1a = a;
        CooCVectorOld final1b = b;
        assertThrows(Exception.class, ()->final1a.setRow(final1b, 1));

        // ---------------------  Sub-case 6 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.15039+0.14454i"), new CNumber("0.62866+0.89828i"), new CNumber("0.74706+0.58685i"), new CNumber("0.21896+0.25363i")};
        aRowIndices = new int[]{0, 1, 2, 2};
        aColIndices = new int[]{1, 4, 1, 2};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5);
        bEntries = new CNumber[]{new CNumber("0.09009+0.73789i"), new CNumber("0.07688+0.38056i")};
        bindices = new int[]{2, 4};
        b = new CooCVectorOld(bShape.get(0), bEntries, bindices);

        CooCMatrixOld final2a = a;
        CooCVectorOld final2b = b;
        assertThrows(Exception.class, ()->final2a.setRow(final2b, -1));

        // ---------------------  Sub-case 7 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.13338+0.01839i"), new CNumber("0.93545+0.80313i"), new CNumber("0.43471+0.42413i"), new CNumber("0.09967+0.38018i")};
        aRowIndices = new int[]{0, 0, 1, 2};
        aColIndices = new int[]{2, 4, 2, 1};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5);
        bEntries = new CNumber[]{new CNumber("0.14938+0.44828i"), new CNumber("0.3092+0.80895i")};
        bindices = new int[]{1, 3};
        b = new CooCVectorOld(bShape.get(0), bEntries, bindices);

        CooCMatrixOld final3a = a;
        CooCVectorOld final3b = b;
        assertThrows(Exception.class, ()->final3a.setRow(final3b, 5));
    }


    @Test
    void setRowDenseTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        CooCMatrixOld a;

        CNumber[] bEntries;
        CVectorOld b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        CooCMatrixOld exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.24854+0.52684i"), new CNumber("0.04226+0.54054i"), new CNumber("0.37425+0.2214i")};
        aRowIndices = new int[]{0, 2, 4};
        aColIndices = new int[]{1, 2, 2};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[]{new CNumber("0.18274+0.25077i"), new CNumber("0.66126+0.89279i"), new CNumber("0.20345+0.58184i")};
        b = new CVectorOld(bEntries);

        expShape = new Shape(5, 3);
        expEntries = new CNumber[]{new CNumber("0.18274+0.25077i"), new CNumber("0.66126+0.89279i"), new CNumber("0.20345+0.58184i"), new CNumber("0.04226+0.54054i"), new CNumber("0.37425+0.2214i")};
        expRowIndices = new int[]{0, 0, 0, 2, 4};
        expColIndices = new int[]{0, 1, 2, 2, 2};
        exp = new CooCMatrixOld(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setRow(b, 0));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(23, 11);
        aEntries = new CNumber[]{new CNumber("0.24012+0.25408i"), new CNumber("0.94941+0.42439i"), new CNumber("0.60769+0.43935i"), new CNumber("0.17206+0.29554i"), new CNumber("0.63172+0.69064i")};
        aRowIndices = new int[]{10, 12, 12, 20, 20};
        aColIndices = new int[]{1, 2, 8, 3, 8};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[]{new CNumber("0.07138+0.45167i"), new CNumber("0.93062+0.14177i"), new CNumber("0.70112+0.41051i"), new CNumber("0.69003+0.96802i"), new CNumber("0.72455+0.53894i"), new CNumber("0.34266+0.33402i"), new CNumber("0.33053+0.27074i"), new CNumber("0.33903+0.43509i"), new CNumber("0.31602+0.3136i"), new CNumber("0.54317+0.89294i"), new CNumber("0.4619+0.62924i")};
        b = new CVectorOld(bEntries);

        expShape = new Shape(23, 11);
        expEntries = new CNumber[]{new CNumber("0.07138+0.45167i"), new CNumber("0.93062+0.14177i"), new CNumber("0.70112+0.41051i"), new CNumber("0.69003+0.96802i"), new CNumber("0.72455+0.53894i"), new CNumber("0.34266+0.33402i"), new CNumber("0.33053+0.27074i"), new CNumber("0.33903+0.43509i"), new CNumber("0.31602+0.3136i"), new CNumber("0.54317+0.89294i"), new CNumber("0.4619+0.62924i"), new CNumber("0.24012+0.25408i"), new CNumber("0.94941+0.42439i"), new CNumber("0.60769+0.43935i"), new CNumber("0.17206+0.29554i"), new CNumber("0.63172+0.69064i")};
        expRowIndices = new int[]{8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 10, 12, 12, 20, 20};
        expColIndices = new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 8, 3, 8};
        exp = new CooCMatrixOld(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setRow(b, 8));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(1000, 5);
        aEntries = new CNumber[]{new CNumber("0.50868+0.78206i"), new CNumber("0.76839+0.55265i"), new CNumber("0.23793+0.45364i"), new CNumber("0.6038+0.00136i"), new CNumber("0.92526+0.6271i"), new CNumber("0.85971+0.35043i"), new CNumber("0.4214+0.88768i"), new CNumber("0.4893+0.68112i"), new CNumber("0.23015+0.89139i")};
        aRowIndices = new int[]{91, 161, 343, 388, 410, 493, 501, 536, 671};
        aColIndices = new int[]{2, 2, 2, 2, 1, 4, 1, 3, 1};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[]{new CNumber("0.80144+0.96716i"), new CNumber("0.03844+0.93879i"), new CNumber("0.30832+0.87148i"), new CNumber("0.01687+0.61229i"), new CNumber("0.13748+0.15688i")};
        b = new CVectorOld(bEntries);

        expShape = new Shape(1000, 5);
        expEntries = new CNumber[]{new CNumber("0.50868+0.78206i"), new CNumber("0.76839+0.55265i"), new CNumber("0.23793+0.45364i"), new CNumber("0.6038+0.00136i"), new CNumber("0.92526+0.6271i"), new CNumber("0.85971+0.35043i"), new CNumber("0.4214+0.88768i"), new CNumber("0.4893+0.68112i"), new CNumber("0.23015+0.89139i"), new CNumber("0.80144+0.96716i"), new CNumber("0.03844+0.93879i"), new CNumber("0.30832+0.87148i"), new CNumber("0.01687+0.61229i"), new CNumber("0.13748+0.15688i")};
        expRowIndices = new int[]{91, 161, 343, 388, 410, 493, 501, 536, 671, 999, 999, 999, 999, 999};
        expColIndices = new int[]{2, 2, 2, 2, 1, 4, 1, 3, 1, 0, 1, 2, 3, 4};
        exp = new CooCMatrixOld(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setRow(b, 999));

        // ---------------------  Sub-case 4 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.71421+0.28634i"), new CNumber("0.52502+0.4433i"), new CNumber("0.1073+0.45825i"), new CNumber("0.8693+0.38204i")};
        aRowIndices = new int[]{1, 1, 2, 2};
        aColIndices = new int[]{1, 4, 1, 3};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[]{new CNumber("0.95282+0.52857i"), new CNumber("0.36597+0.81948i"), new CNumber("0.88462+0.30582i"), new CNumber("0.52975+0.47877i")};
        b = new CVectorOld(bEntries);

        CooCMatrixOld final0a = a;
        CVectorOld final0b = b;
        assertThrows(Exception.class, ()->final0a.setRow(final0b, 2));

        // ---------------------  Sub-case 5 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.29379+0.9404i"), new CNumber("0.5153+0.20832i"), new CNumber("0.06228+0.70059i"), new CNumber("0.83618+0.51948i")};
        aRowIndices = new int[]{0, 0, 1, 2};
        aColIndices = new int[]{1, 3, 1, 0};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[]{new CNumber("0.33926+0.68357i"), new CNumber("0.458+0.41036i"), new CNumber("0.77024+0.46695i"), new CNumber("0.37632+0.2523i"), new CNumber("0.05953+0.70822i"), new CNumber("0.83693+0.9951i")};
        b = new CVectorOld(bEntries);

        CooCMatrixOld final1a = a;
        CVectorOld final1b = b;
        assertThrows(Exception.class, ()->final1a.setRow(final1b, 1));

        // ---------------------  Sub-case 6 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.73954+0.68349i"), new CNumber("0.06269+0.10072i"), new CNumber("0.12821+0.83691i"), new CNumber("0.20449+0.81254i")};
        aRowIndices = new int[]{0, 1, 2, 2};
        aColIndices = new int[]{0, 2, 1, 2};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[]{new CNumber("0.67277+0.3974i"), new CNumber("0.3877+0.35631i"), new CNumber("0.03545+0.36513i"), new CNumber("0.43889+0.47356i"), new CNumber("0.99912+0.69817i")};
        b = new CVectorOld(bEntries);

        CooCMatrixOld final2a = a;
        CVectorOld final2b = b;
        assertThrows(Exception.class, ()->final2a.setRow(final2b, -1));

        // ---------------------  Sub-case 7 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.62601+0.61241i"), new CNumber("0.21243+0.06131i"), new CNumber("0.61868+0.93782i"), new CNumber("0.24115+0.58141i")};
        aRowIndices = new int[]{0, 0, 1, 2};
        aColIndices = new int[]{1, 2, 0, 3};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[]{new CNumber("0.50239+0.50692i"), new CNumber("0.21647+0.90742i"), new CNumber("0.71632+0.59001i"), new CNumber("0.96968+0.70681i"), new CNumber("0.8206+0.57191i")};
        b = new CVectorOld(bEntries);

        CooCMatrixOld final3a = a;
        CVectorOld final3b = b;
        assertThrows(Exception.class, ()->final3a.setRow(final3b, 5));
    }


    @Test
    void setRowDenseArrayTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        CooCMatrixOld a;

        CNumber[] b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        CooCMatrixOld exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.24854+0.52684i"), new CNumber("0.04226+0.54054i"), new CNumber("0.37425+0.2214i")};
        aRowIndices = new int[]{0, 2, 4};
        aColIndices = new int[]{1, 2, 2};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        b = new CNumber[]{new CNumber("0.18274+0.25077i"), new CNumber("0.66126+0.89279i"), new CNumber("0.20345+0.58184i")};

        expShape = new Shape(5, 3);
        expEntries = new CNumber[]{new CNumber("0.18274+0.25077i"), new CNumber("0.66126+0.89279i"), new CNumber("0.20345+0.58184i"), new CNumber("0.04226+0.54054i"), new CNumber("0.37425+0.2214i")};
        expRowIndices = new int[]{0, 0, 0, 2, 4};
        expColIndices = new int[]{0, 1, 2, 2, 2};
        exp = new CooCMatrixOld(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setRow(b, 0));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(23, 11);
        aEntries = new CNumber[]{new CNumber("0.24012+0.25408i"), new CNumber("0.94941+0.42439i"), new CNumber("0.60769+0.43935i"), new CNumber("0.17206+0.29554i"), new CNumber("0.63172+0.69064i")};
        aRowIndices = new int[]{10, 12, 12, 20, 20};
        aColIndices = new int[]{1, 2, 8, 3, 8};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        b = new CNumber[]{new CNumber("0.07138+0.45167i"), new CNumber("0.93062+0.14177i"), new CNumber("0.70112+0.41051i"), new CNumber("0.69003+0.96802i"), new CNumber("0.72455+0.53894i"), new CNumber("0.34266+0.33402i"), new CNumber("0.33053+0.27074i"), new CNumber("0.33903+0.43509i"), new CNumber("0.31602+0.3136i"), new CNumber("0.54317+0.89294i"), new CNumber("0.4619+0.62924i")};

        expShape = new Shape(23, 11);
        expEntries = new CNumber[]{new CNumber("0.07138+0.45167i"), new CNumber("0.93062+0.14177i"), new CNumber("0.70112+0.41051i"), new CNumber("0.69003+0.96802i"), new CNumber("0.72455+0.53894i"), new CNumber("0.34266+0.33402i"), new CNumber("0.33053+0.27074i"), new CNumber("0.33903+0.43509i"), new CNumber("0.31602+0.3136i"), new CNumber("0.54317+0.89294i"), new CNumber("0.4619+0.62924i"), new CNumber("0.24012+0.25408i"), new CNumber("0.94941+0.42439i"), new CNumber("0.60769+0.43935i"), new CNumber("0.17206+0.29554i"), new CNumber("0.63172+0.69064i")};
        expRowIndices = new int[]{8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 10, 12, 12, 20, 20};
        expColIndices = new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 8, 3, 8};
        exp = new CooCMatrixOld(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setRow(b, 8));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(1000, 5);
        aEntries = new CNumber[]{new CNumber("0.50868+0.78206i"), new CNumber("0.76839+0.55265i"), new CNumber("0.23793+0.45364i"), new CNumber("0.6038+0.00136i"), new CNumber("0.92526+0.6271i"), new CNumber("0.85971+0.35043i"), new CNumber("0.4214+0.88768i"), new CNumber("0.4893+0.68112i"), new CNumber("0.23015+0.89139i")};
        aRowIndices = new int[]{91, 161, 343, 388, 410, 493, 501, 536, 671};
        aColIndices = new int[]{2, 2, 2, 2, 1, 4, 1, 3, 1};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        b = new CNumber[]{new CNumber("0.80144+0.96716i"), new CNumber("0.03844+0.93879i"), new CNumber("0.30832+0.87148i"), new CNumber("0.01687+0.61229i"), new CNumber("0.13748+0.15688i")};

        expShape = new Shape(1000, 5);
        expEntries = new CNumber[]{new CNumber("0.50868+0.78206i"), new CNumber("0.76839+0.55265i"), new CNumber("0.23793+0.45364i"), new CNumber("0.6038+0.00136i"), new CNumber("0.92526+0.6271i"), new CNumber("0.85971+0.35043i"), new CNumber("0.4214+0.88768i"), new CNumber("0.4893+0.68112i"), new CNumber("0.23015+0.89139i"), new CNumber("0.80144+0.96716i"), new CNumber("0.03844+0.93879i"), new CNumber("0.30832+0.87148i"), new CNumber("0.01687+0.61229i"), new CNumber("0.13748+0.15688i")};
        expRowIndices = new int[]{91, 161, 343, 388, 410, 493, 501, 536, 671, 999, 999, 999, 999, 999};
        expColIndices = new int[]{2, 2, 2, 2, 1, 4, 1, 3, 1, 0, 1, 2, 3, 4};
        exp = new CooCMatrixOld(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setRow(b, 999));

        // ---------------------  Sub-case 4 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.71421+0.28634i"), new CNumber("0.52502+0.4433i"), new CNumber("0.1073+0.45825i"), new CNumber("0.8693+0.38204i")};
        aRowIndices = new int[]{1, 1, 2, 2};
        aColIndices = new int[]{1, 4, 1, 3};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        b = new CNumber[]{new CNumber("0.95282+0.52857i"), new CNumber("0.36597+0.81948i"), new CNumber("0.88462+0.30582i"), new CNumber("0.52975+0.47877i")};

        CooCMatrixOld final0a = a;
        CNumber[] final0b = b;
        assertThrows(Exception.class, ()->final0a.setRow(final0b, 2));

        // ---------------------  Sub-case 5 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.29379+0.9404i"), new CNumber("0.5153+0.20832i"), new CNumber("0.06228+0.70059i"), new CNumber("0.83618+0.51948i")};
        aRowIndices = new int[]{0, 0, 1, 2};
        aColIndices = new int[]{1, 3, 1, 0};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        b = new CNumber[]{new CNumber("0.33926+0.68357i"), new CNumber("0.458+0.41036i"), new CNumber("0.77024+0.46695i"), new CNumber("0.37632+0.2523i"), new CNumber("0.05953+0.70822i"), new CNumber("0.83693+0.9951i")};

        CooCMatrixOld final1a = a;
        CNumber[] final1b = b;
        assertThrows(Exception.class, ()->final1a.setRow(final1b, 1));

        // ---------------------  Sub-case 6 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.73954+0.68349i"), new CNumber("0.06269+0.10072i"), new CNumber("0.12821+0.83691i"), new CNumber("0.20449+0.81254i")};
        aRowIndices = new int[]{0, 1, 2, 2};
        aColIndices = new int[]{0, 2, 1, 2};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        b = new CNumber[]{new CNumber("0.67277+0.3974i"), new CNumber("0.3877+0.35631i"), new CNumber("0.03545+0.36513i"), new CNumber("0.43889+0.47356i"), new CNumber("0.99912+0.69817i")};

        CooCMatrixOld final2a = a;
        CNumber[] final2b = b;
        assertThrows(Exception.class, ()->final2a.setRow(final2b, -1));

        // ---------------------  Sub-case 7 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.62601+0.61241i"), new CNumber("0.21243+0.06131i"), new CNumber("0.61868+0.93782i"), new CNumber("0.24115+0.58141i")};
        aRowIndices = new int[]{0, 0, 1, 2};
        aColIndices = new int[]{1, 2, 0, 3};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        b = new CNumber[]{new CNumber("0.50239+0.50692i"), new CNumber("0.21647+0.90742i"), new CNumber("0.71632+0.59001i"), new CNumber("0.96968+0.70681i"), new CNumber("0.8206+0.57191i")};

        CooCMatrixOld final3a = a;
        CNumber[] final3b = b;
        assertThrows(Exception.class, ()->final3a.setRow(final3b, 5));
    }


    @Test
    void setRowRealDenseTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        CooCMatrixOld a;

        double[] b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        CooCMatrixOld exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.69244+0.83605i"), new CNumber("0.33157+0.97127i"), new CNumber("0.70234+0.33309i")};
        aRowIndices = new int[]{0, 1, 3};
        aColIndices = new int[]{1, 2, 1};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        b = new double[]{0.62031, 0.41865, 0.18833};

        expShape = new Shape(5, 3);
        expEntries = new CNumber[]{new CNumber("0.62031"), new CNumber("0.41865"), new CNumber("0.18833"), new CNumber("0.33157+0.97127i"), new CNumber("0.70234+0.33309i")};
        expRowIndices = new int[]{0, 0, 0, 1, 3};
        expColIndices = new int[]{0, 1, 2, 2, 1};
        exp = new CooCMatrixOld(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setRow(b, 0));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(23, 11);
        aEntries = new CNumber[]{new CNumber("0.75052+0.25582i"), new CNumber("0.02637+0.05068i"), new CNumber("0.59336+0.89595i"), new CNumber("0.62881+0.6464i"), new CNumber("0.13137+0.62554i")};
        aRowIndices = new int[]{1, 9, 10, 12, 16};
        aColIndices = new int[]{5, 10, 9, 10, 4};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        b = new double[]{0.04229, 0.56986, 0.79249, 0.25465, 0.3391, 0.90516, 0.11465, 0.59902, 0.56749, 0.14721, 0.68973};

        expShape = new Shape(23, 11);
        expEntries = new CNumber[]{new CNumber("0.75052+0.25582i"), new CNumber("0.04229"), new CNumber("0.56986"), new CNumber("0.79249"), new CNumber("0.25465"), new CNumber("0.3391"), new CNumber("0.90516"), new CNumber("0.11465"), new CNumber("0.59902"), new CNumber("0.56749"), new CNumber("0.14721"), new CNumber("0.68973"), new CNumber("0.02637+0.05068i"), new CNumber("0.59336+0.89595i"), new CNumber("0.62881+0.6464i"), new CNumber("0.13137+0.62554i")};
        expRowIndices = new int[]{1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 10, 12, 16};
        expColIndices = new int[]{5, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 10, 4};
        exp = new CooCMatrixOld(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setRow(b, 8));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(1000, 5);
        aEntries = new CNumber[]{new CNumber("0.39021+0.98248i"), new CNumber("0.41224+0.881i"), new CNumber("0.28586+0.94809i"), new CNumber("0.28376+0.59994i"), new CNumber("0.3929+0.31445i"), new CNumber("0.23057+0.43094i"), new CNumber("0.98145+0.91882i"), new CNumber("0.31723+0.92016i"), new CNumber("0.02067+0.63208i")};
        aRowIndices = new int[]{28, 108, 242, 402, 462, 540, 551, 655, 857};
        aColIndices = new int[]{4, 1, 2, 3, 4, 3, 1, 3, 3};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        b = new double[]{0.2209, 0.15271, 0.88025, 0.53016, 0.35227};

        expShape = new Shape(1000, 5);
        expEntries = new CNumber[]{new CNumber("0.39021+0.98248i"), new CNumber("0.41224+0.881i"), new CNumber("0.28586+0.94809i"), new CNumber("0.28376+0.59994i"), new CNumber("0.3929+0.31445i"), new CNumber("0.23057+0.43094i"), new CNumber("0.98145+0.91882i"), new CNumber("0.31723+0.92016i"), new CNumber("0.02067+0.63208i"), new CNumber("0.2209"), new CNumber("0.15271"), new CNumber("0.88025"), new CNumber("0.53016"), new CNumber("0.35227")};
        expRowIndices = new int[]{28, 108, 242, 402, 462, 540, 551, 655, 857, 999, 999, 999, 999, 999};
        expColIndices = new int[]{4, 1, 2, 3, 4, 3, 1, 3, 3, 0, 1, 2, 3, 4};
        exp = new CooCMatrixOld(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setRow(b, 999));

        // ---------------------  Sub-case 4 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.04294+0.18694i"), new CNumber("0.52283+0.8394i"), new CNumber("0.6545+0.75458i"), new CNumber("0.5127+0.78475i")};
        aRowIndices = new int[]{0, 0, 0, 1};
        aColIndices = new int[]{0, 2, 4, 2};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        b = new double[]{0.8634, 0.34815, 0.10091, 0.97066};

        CooCMatrixOld final0a = a;
        double[] final0b = b;
        assertThrows(Exception.class, ()->final0a.setRow(final0b, 2));

        // ---------------------  Sub-case 5 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.31331+0.03353i"), new CNumber("0.51581+0.19996i"), new CNumber("0.38324+0.26027i"), new CNumber("0.37248+0.18355i")};
        aRowIndices = new int[]{0, 0, 1, 1};
        aColIndices = new int[]{1, 3, 0, 1};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        b = new double[]{0.5295, 0.98271, 0.08481, 0.72014, 0.8882, 0.51333};

        CooCMatrixOld final1a = a;
        double[] final1b = b;
        assertThrows(Exception.class, ()->final1a.setRow(final1b, 1));

        // ---------------------  Sub-case 6 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.86105+0.45926i"), new CNumber("0.37429+0.81014i"), new CNumber("0.36859+0.2332i"), new CNumber("0.27918+0.26377i")};
        aRowIndices = new int[]{0, 1, 2, 2};
        aColIndices = new int[]{3, 2, 0, 3};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        b = new double[]{0.2129, 0.57516, 0.24863, 0.86802, 0.15185};

        CooCMatrixOld final2a = a;
        double[] final2b = b;
        assertThrows(Exception.class, ()->final2a.setRow(final2b, -1));

        // ---------------------  Sub-case 7 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.29679+0.87419i"), new CNumber("0.4692+0.28999i"), new CNumber("0.33056+0.82327i"), new CNumber("0.71285+0.75284i")};
        aRowIndices = new int[]{0, 0, 1, 2};
        aColIndices = new int[]{2, 3, 1, 1};
        a = new CooCMatrixOld(aShape, aEntries, aRowIndices, aColIndices);

        b = new double[]{0.16499, 0.08612, 0.76084, 0.36419, 0.45576};

        CooCMatrixOld final3a = a;
        double[] final3b = b;
        assertThrows(Exception.class, ()->final3a.setRow(final3b, 5));
    }
}
