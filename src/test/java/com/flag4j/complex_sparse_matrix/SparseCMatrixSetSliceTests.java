package com.flag4j.complex_sparse_matrix;

import com.flag4j.Matrix;
import com.flag4j.Shape;
import com.flag4j.SparseCMatrix;
import com.flag4j.SparseMatrix;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class SparseCMatrixSetSliceTests {

    @Test
    void setSliceTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        SparseCMatrix a;

        Shape bShape;
        int[] bRowIndices;
        int[] bColIndices;
        CNumber[] bEntries;
        SparseCMatrix b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        SparseCMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.04651+0.52949i"), new CNumber("0.81971+0.45861i"), new CNumber("0.51405+0.44347i")};
        aRowIndices = new int[]{0, 1, 4};
        aColIndices = new int[]{2, 0, 1};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(2, 3);
        bEntries = new CNumber[]{new CNumber("0.28262+0.26286i"), new CNumber("0.48758+0.68001i")};
        bRowIndices = new int[]{0, 1};
        bColIndices = new int[]{1, 0};
        b = new SparseCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(5, 3);
        expEntries = new CNumber[]{new CNumber("0.04651+0.52949i"), new CNumber("0.81971+0.45861i"), new CNumber("0.28262+0.26286i"), new CNumber("0.48758+0.68001i"), new CNumber("0.51405+0.44347i")};
        expRowIndices = new int[]{0, 1, 2, 3, 4};
        expColIndices = new int[]{2, 0, 1, 0, 1};
        exp = new SparseCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setSlice(b, 2, 0));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(23, 11);
        aEntries = new CNumber[]{new CNumber("0.74234+0.12429i"), new CNumber("0.64073+0.54402i"), new CNumber("0.11094+0.92024i"), new CNumber("0.33648+0.68645i"), new CNumber("0.91835+0.17975i")};
        aRowIndices = new int[]{0, 17, 18, 18, 22};
        aColIndices = new int[]{9, 4, 7, 8, 6};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 9);
        bEntries = new CNumber[]{new CNumber("0.09072+0.3904i"), new CNumber("0.63089+0.4616i"), new CNumber("0.24842+0.06922i"), new CNumber("0.54202+0.29573i"), new CNumber("0.53712+0.69814i"), new CNumber("0.10022+0.89626i"), new CNumber("0.65279+0.93909i"), new CNumber("0.02023+0.9508i"), new CNumber("0.4347+0.59478i"), new CNumber("0.36622+0.39646i"), new CNumber("0.50713+0.84846i"), new CNumber("0.60974+0.76682i"), new CNumber("0.65967+0.19455i"), new CNumber("0.54107+0.91789i"), new CNumber("0.50234+0.53841i"), new CNumber("0.46215+0.81439i")};
        bRowIndices = new int[]{0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4};
        bColIndices = new int[]{1, 3, 5, 7, 1, 3, 5, 8, 1, 2, 6, 7, 4, 6, 8, 3};
        b = new SparseCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(23, 11);
        expEntries = new CNumber[]{new CNumber("0.74234+0.12429i"), new CNumber("0.64073+0.54402i"), new CNumber("0.09072+0.3904i"), new CNumber("0.63089+0.4616i"), new CNumber("0.24842+0.06922i"), new CNumber("0.54202+0.29573i"), new CNumber("0.53712+0.69814i"), new CNumber("0.10022+0.89626i"), new CNumber("0.65279+0.93909i"), new CNumber("0.02023+0.9508i"), new CNumber("0.4347+0.59478i"), new CNumber("0.36622+0.39646i"), new CNumber("0.50713+0.84846i"), new CNumber("0.60974+0.76682i"), new CNumber("0.65967+0.19455i"), new CNumber("0.54107+0.91789i"), new CNumber("0.50234+0.53841i"), new CNumber("0.46215+0.81439i")};
        expRowIndices = new int[]{0, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 21, 22};
        expColIndices = new int[]{9, 4, 2, 4, 6, 8, 2, 4, 6, 9, 2, 3, 7, 8, 5, 7, 9, 4};
        exp = new SparseCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setSlice(b, 18, 1));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(1000, 5);
        aEntries = new CNumber[]{new CNumber("0.63959+0.23626i"), new CNumber("0.20705+0.66846i"), new CNumber("0.67522+0.61482i"), new CNumber("0.66809+0.71426i"), new CNumber("0.12147+0.6442i"), new CNumber("0.41783+0.33737i"), new CNumber("0.80593+0.83277i"), new CNumber("0.90324+0.45023i"), new CNumber("0.95079+0.52534i")};
        aRowIndices = new int[]{51, 53, 69, 387, 399, 408, 683, 899, 998};
        aColIndices = new int[]{4, 4, 1, 4, 0, 3, 4, 0, 4};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3, 2);
        bEntries = new CNumber[]{new CNumber("0.17744+0.9073i"), new CNumber("0.30357+0.46591i")};
        bRowIndices = new int[]{0, 2};
        bColIndices = new int[]{1, 0};
        b = new SparseCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(1000, 5);
        expEntries = new CNumber[]{new CNumber("0.17744+0.9073i"), new CNumber("0.30357+0.46591i"), new CNumber("0.63959+0.23626i"), new CNumber("0.20705+0.66846i"), new CNumber("0.67522+0.61482i"), new CNumber("0.66809+0.71426i"), new CNumber("0.12147+0.6442i"), new CNumber("0.41783+0.33737i"), new CNumber("0.80593+0.83277i"), new CNumber("0.90324+0.45023i"), new CNumber("0.95079+0.52534i")};
        expRowIndices = new int[]{0, 2, 51, 53, 69, 387, 399, 408, 683, 899, 998};
        expColIndices = new int[]{1, 0, 4, 4, 1, 4, 0, 3, 4, 0, 4};
        exp = new SparseCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setSlice(b, 0, 0));

        // ---------------------  Sub-case 4 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.0623+0.17801i"), new CNumber("0.20499+0.08446i"), new CNumber("0.40636+0.24712i"), new CNumber("0.39867+0.51485i")};
        aRowIndices = new int[]{0, 0, 1, 2};
        aColIndices = new int[]{1, 3, 0, 1};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(1, 3);
        bEntries = new CNumber[]{new CNumber("0.31142+0.70957i")};
        bRowIndices = new int[]{0};
        bColIndices = new int[]{2};
        b = new SparseCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        SparseCMatrix final0a = a;
        SparseCMatrix final0b = b;
        assertThrows(Exception.class, ()->final0a.setSlice(final0b, -1, 2));

        // ---------------------  Sub-case 5 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.53707+0.34618i"), new CNumber("0.75684+0.20186i"), new CNumber("0.33387+0.5436i"), new CNumber("0.35621+0.29147i")};
        aRowIndices = new int[]{0, 0, 0, 1};
        aColIndices = new int[]{0, 1, 3, 3};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3, 1);
        bEntries = new CNumber[]{new CNumber("0.14592+0.87419i")};
        bRowIndices = new int[]{0};
        bColIndices = new int[]{0};
        b = new SparseCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        SparseCMatrix final1a = a;
        SparseCMatrix final1b = b;
        assertThrows(Exception.class, ()->final1a.setSlice(final1b, 0, 16));

        // ---------------------  Sub-case 6 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.64684+0.98603i"), new CNumber("0.00028+0.59363i"), new CNumber("0.46343+0.5909i"), new CNumber("0.20891+0.6038i")};
        aRowIndices = new int[]{0, 1, 2, 2};
        aColIndices = new int[]{1, 0, 0, 4};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(2, 2);
        bEntries = new CNumber[]{new CNumber("0.17692+0.17537i")};
        bRowIndices = new int[]{0};
        bColIndices = new int[]{1};
        b = new SparseCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        SparseCMatrix final2a = a;
        SparseCMatrix final2b = b;
        assertThrows(Exception.class, ()->final2a.setSlice(final2b, 2, 0));

        // ---------------------  Sub-case 7 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.39986+0.08396i"), new CNumber("0.26114+0.5311i"), new CNumber("0.76285+0.27602i"), new CNumber("0.82919+0.08939i")};
        aRowIndices = new int[]{0, 0, 1, 2};
        aColIndices = new int[]{1, 4, 0, 0};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(1, 3);
        bEntries = new CNumber[]{new CNumber("0.57959+0.82344i")};
        bRowIndices = new int[]{0};
        bColIndices = new int[]{1};
        b = new SparseCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        SparseCMatrix final3a = a;
        SparseCMatrix final3b = b;
        assertThrows(Exception.class, ()->final3a.setSlice(final3b, 0, 4));
    }


    @Test
    void setSliceRealSparseTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        SparseCMatrix a;

        Shape bShape;
        int[] bRowIndices;
        int[] bColIndices;
        double[] bEntries;
        SparseMatrix b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        SparseCMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.68534+0.15364i"), new CNumber("0.12931+0.36459i"), new CNumber("0.70074+0.51258i")};
        aRowIndices = new int[]{2, 2, 4};
        aColIndices = new int[]{0, 2, 2};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(2, 3);
        bEntries = new double[]{0.27432, 0.15004};
        bRowIndices = new int[]{0, 1};
        bColIndices = new int[]{2, 2};
        b = new SparseMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(5, 3);
        expEntries = new CNumber[]{new CNumber("0.27432"), new CNumber("0.15004"), new CNumber("0.70074+0.51258i")};
        expRowIndices = new int[]{2, 3, 4};
        expColIndices = new int[]{2, 2, 2};
        exp = new SparseCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setSlice(b, 2, 0));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(23, 11);
        aEntries = new CNumber[]{new CNumber("0.16362+0.66728i"), new CNumber("0.64264+0.60954i"), new CNumber("0.84172+0.72746i"), new CNumber("0.58218+0.47918i"), new CNumber("0.29901+0.95937i")};
        aRowIndices = new int[]{4, 7, 8, 11, 15};
        aColIndices = new int[]{8, 9, 5, 0, 10};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 9);
        bEntries = new double[]{0.29718, 0.96359, 0.33835, 0.789, 0.71905, 0.99707, 0.65239, 0.3993, 0.50486, 0.63918, 0.704, 0.81702, 0.68953, 0.78047, 0.1208, 0.87019};
        bRowIndices = new int[]{0, 0, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4};
        bColIndices = new int[]{0, 6, 7, 0, 3, 4, 5, 6, 8, 0, 1, 3, 6, 7, 8, 5};
        b = new SparseMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(23, 11);
        expEntries = new CNumber[]{new CNumber("0.16362+0.66728i"), new CNumber("0.64264+0.60954i"), new CNumber("0.84172+0.72746i"), new CNumber("0.58218+0.47918i"), new CNumber("0.29901+0.95937i"), new CNumber("0.29718"), new CNumber("0.96359"), new CNumber("0.33835"), new CNumber("0.789"), new CNumber("0.71905"), new CNumber("0.99707"), new CNumber("0.65239"), new CNumber("0.3993"), new CNumber("0.50486"), new CNumber("0.63918"), new CNumber("0.704"), new CNumber("0.81702"), new CNumber("0.68953"), new CNumber("0.78047"), new CNumber("0.1208"), new CNumber("0.87019")};
        expRowIndices = new int[]{4, 7, 8, 11, 15, 18, 18, 19, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 22};
        expColIndices = new int[]{8, 9, 5, 0, 10, 1, 7, 8, 1, 4, 5, 6, 7, 9, 1, 2, 4, 7, 8, 9, 6};
        exp = new SparseCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setSlice(b, 18, 1));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(1000, 5);
        aEntries = new CNumber[]{new CNumber("0.79025+0.48916i"), new CNumber("0.01304+0.7221i"), new CNumber("0.57511+0.29405i"), new CNumber("0.14956+0.81435i"), new CNumber("0.79801+0.77858i"), new CNumber("0.8572+0.96002i"), new CNumber("0.95768+0.68413i"), new CNumber("0.95904+0.14831i"), new CNumber("0.77543+0.11693i")};
        aRowIndices = new int[]{211, 216, 320, 333, 529, 549, 591, 699, 792};
        aColIndices = new int[]{0, 2, 1, 2, 4, 3, 0, 3, 3};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3, 2);
        bEntries = new double[]{0.51439, 0.03478};
        bRowIndices = new int[]{0, 2};
        bColIndices = new int[]{0, 0};
        b = new SparseMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(1000, 5);
        expEntries = new CNumber[]{new CNumber("0.51439"), new CNumber("0.03478"), new CNumber("0.79025+0.48916i"), new CNumber("0.01304+0.7221i"), new CNumber("0.57511+0.29405i"), new CNumber("0.14956+0.81435i"), new CNumber("0.79801+0.77858i"), new CNumber("0.8572+0.96002i"), new CNumber("0.95768+0.68413i"), new CNumber("0.95904+0.14831i"), new CNumber("0.77543+0.11693i")};
        expRowIndices = new int[]{0, 2, 211, 216, 320, 333, 529, 549, 591, 699, 792};
        expColIndices = new int[]{0, 0, 0, 2, 1, 2, 4, 3, 0, 3, 3};
        exp = new SparseCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setSlice(b, 0, 0));

        // ---------------------  Sub-case 4 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.24662+0.81963i"), new CNumber("0.40546+0.4195i"), new CNumber("0.16365+0.77829i"), new CNumber("0.0649+0.63156i")};
        aRowIndices = new int[]{0, 0, 2, 2};
        aColIndices = new int[]{1, 4, 0, 3};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(1, 3);
        bEntries = new double[]{0.18013};
        bRowIndices = new int[]{0};
        bColIndices = new int[]{1};
        b = new SparseMatrix(bShape, bEntries, bRowIndices, bColIndices);

        SparseCMatrix final0a = a;
        SparseMatrix final0b = b;
        assertThrows(Exception.class, ()->final0a.setSlice(final0b, -1, 2));

        // ---------------------  Sub-case 5 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.21205+0.75748i"), new CNumber("0.35975+0.62959i"), new CNumber("0.62822+0.27063i"), new CNumber("0.16095+0.53217i")};
        aRowIndices = new int[]{0, 1, 2, 2};
        aColIndices = new int[]{3, 3, 0, 1};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3, 1);
        bEntries = new double[]{0.22882};
        bRowIndices = new int[]{0};
        bColIndices = new int[]{0};
        b = new SparseMatrix(bShape, bEntries, bRowIndices, bColIndices);

        SparseCMatrix final1a = a;
        SparseMatrix final1b = b;
        assertThrows(Exception.class, ()->final1a.setSlice(final1b, 0, 16));

        // ---------------------  Sub-case 6 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.46767+0.26761i"), new CNumber("0.45674+0.03671i"), new CNumber("0.56813+0.01382i"), new CNumber("0.04182+0.30644i")};
        aRowIndices = new int[]{1, 1, 2, 2};
        aColIndices = new int[]{0, 2, 2, 4};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(2, 2);
        bEntries = new double[]{0.09822};
        bRowIndices = new int[]{0};
        bColIndices = new int[]{1};
        b = new SparseMatrix(bShape, bEntries, bRowIndices, bColIndices);

        SparseCMatrix final2a = a;
        SparseMatrix final2b = b;
        assertThrows(Exception.class, ()->final2a.setSlice(final2b, 2, 0));

        // ---------------------  Sub-case 7 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.94869+0.51293i"), new CNumber("0.54126+0.50489i"), new CNumber("0.42347+0.40127i"), new CNumber("0.51752+0.23141i")};
        aRowIndices = new int[]{1, 1, 2, 2};
        aColIndices = new int[]{0, 1, 1, 4};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(1, 3);
        bEntries = new double[]{0.51567};
        bRowIndices = new int[]{0};
        bColIndices = new int[]{1};
        b = new SparseMatrix(bShape, bEntries, bRowIndices, bColIndices);

        SparseCMatrix final3a = a;
        SparseMatrix final3b = b;
        assertThrows(Exception.class, ()->final3a.setSlice(final3b, 0, 4));
    }


    @Test
    void setSliceRealDenseTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        SparseCMatrix a;

        double[][] bEntries;
        Matrix b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        SparseCMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.92932+0.83391i"), new CNumber("0.75564+0.4746i"), new CNumber("0.67667+0.02131i")};
        aRowIndices = new int[]{1, 2, 3};
        aColIndices = new int[]{2, 1, 0};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.20641, 0.00627, 0.74659},
                {0.32765, 0.21592, 0.62641}};
        b = new Matrix(bEntries);

        expShape = new Shape(5, 3);
        expEntries = new CNumber[]{new CNumber("0.92932+0.83391i"), new CNumber("0.20641"), new CNumber("0.00627"), new CNumber("0.74659"), new CNumber("0.32765"), new CNumber("0.21592"), new CNumber("0.62641")};
        expRowIndices = new int[]{1, 2, 2, 2, 3, 3, 3};
        expColIndices = new int[]{2, 0, 1, 2, 0, 1, 2};
        exp = new SparseCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setSlice(b, 2, 0));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(23, 11);
        aEntries = new CNumber[]{new CNumber("0.78261+0.81287i"), new CNumber("0.92047+0.1968i"), new CNumber("0.19346+0.66988i"), new CNumber("0.79317+0.5482i"), new CNumber("0.86173+0.26806i")};
        aRowIndices = new int[]{2, 12, 13, 20, 22};
        aColIndices = new int[]{8, 8, 8, 9, 3};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.42509, 0.86635, 0.49669, 0.12935, 0.58896, 0.55769, 0.74153, 0.3902, 0.30427},
                {0.47135, 0.27404, 0.77679, 0.18645, 0.14262, 0.34505, 0.96778, 0.08602, 0.04046},
                {0.17124, 0.48106, 0.81049, 0.98155, 0.92245, 0.22864, 0.24145, 0.44011, 0.50031},
                {0.73361, 0.23513, 0.61501, 0.95234, 0.48264, 0.74098, 0.69689, 0.41605, 0.41617},
                {0.58739, 0.71357, 0.91358, 0.35843, 0.94982, 0.39818, 0.24228, 0.26308, 0.30217}};
        b = new Matrix(bEntries);

        expShape = new Shape(23, 11);
        expEntries = new CNumber[]{new CNumber("0.78261+0.81287i"), new CNumber("0.92047+0.1968i"), new CNumber("0.19346+0.66988i"), new CNumber("0.42509"), new CNumber("0.86635"), new CNumber("0.49669"), new CNumber("0.12935"), new CNumber("0.58896"), new CNumber("0.55769"), new CNumber("0.74153"), new CNumber("0.3902"), new CNumber("0.30427"), new CNumber("0.47135"), new CNumber("0.27404"), new CNumber("0.77679"), new CNumber("0.18645"), new CNumber("0.14262"), new CNumber("0.34505"), new CNumber("0.96778"), new CNumber("0.08602"), new CNumber("0.04046"), new CNumber("0.17124"), new CNumber("0.48106"), new CNumber("0.81049"), new CNumber("0.98155"), new CNumber("0.92245"), new CNumber("0.22864"), new CNumber("0.24145"), new CNumber("0.44011"), new CNumber("0.50031"), new CNumber("0.73361"), new CNumber("0.23513"), new CNumber("0.61501"), new CNumber("0.95234"), new CNumber("0.48264"), new CNumber("0.74098"), new CNumber("0.69689"), new CNumber("0.41605"), new CNumber("0.41617"), new CNumber("0.58739"), new CNumber("0.71357"), new CNumber("0.91358"), new CNumber("0.35843"), new CNumber("0.94982"), new CNumber("0.39818"), new CNumber("0.24228"), new CNumber("0.26308"), new CNumber("0.30217")};
        expRowIndices = new int[]{2, 12, 13, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22};
        expColIndices = new int[]{8, 8, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        exp = new SparseCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setSlice(b, 18, 1));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(1000, 5);
        aEntries = new CNumber[]{new CNumber("0.47086+0.22829i"), new CNumber("0.3198+0.27911i"), new CNumber("0.15627+0.18005i"), new CNumber("0.8154+0.86608i"), new CNumber("0.29655+0.07095i"), new CNumber("0.82134+0.18228i"), new CNumber("0.35707+0.25355i"), new CNumber("0.66031+0.70124i"), new CNumber("0.97115+0.2385i")};
        aRowIndices = new int[]{104, 108, 134, 162, 201, 243, 356, 599, 750};
        aColIndices = new int[]{1, 3, 1, 4, 0, 4, 1, 3, 3};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.37385, 0.16278},
                {0.16053, 0.78368},
                {0.81585, 0.09195}};
        b = new Matrix(bEntries);

        expShape = new Shape(1000, 5);
        expEntries = new CNumber[]{new CNumber("0.37385"), new CNumber("0.16278"), new CNumber("0.16053"), new CNumber("0.78368"), new CNumber("0.81585"), new CNumber("0.09195"), new CNumber("0.47086+0.22829i"), new CNumber("0.3198+0.27911i"), new CNumber("0.15627+0.18005i"), new CNumber("0.8154+0.86608i"), new CNumber("0.29655+0.07095i"), new CNumber("0.82134+0.18228i"), new CNumber("0.35707+0.25355i"), new CNumber("0.66031+0.70124i"), new CNumber("0.97115+0.2385i")};
        expRowIndices = new int[]{0, 0, 1, 1, 2, 2, 104, 108, 134, 162, 201, 243, 356, 599, 750};
        expColIndices = new int[]{0, 1, 0, 1, 0, 1, 1, 3, 1, 4, 0, 4, 1, 3, 3};
        exp = new SparseCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setSlice(b, 0, 0));

        // ---------------------  Sub-case 4 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.76161+0.32703i"), new CNumber("0.86463+0.73067i"), new CNumber("0.65177+0.53146i"), new CNumber("0.23319+0.56897i")};
        aRowIndices = new int[]{1, 2, 2, 2};
        aColIndices = new int[]{0, 1, 2, 4};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.2427, 0.27463, 0.00804}};
        b = new Matrix(bEntries);

        SparseCMatrix final0a = a;
        Matrix final0b = b;
        assertThrows(Exception.class, ()->final0a.setSlice(final0b, -1, 2));

        // ---------------------  Sub-case 5 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.41438+0.13745i"), new CNumber("0.62856+0.69366i"), new CNumber("0.40666+0.4999i"), new CNumber("0.81274+0.10541i")};
        aRowIndices = new int[]{0, 0, 1, 2};
        aColIndices = new int[]{1, 4, 3, 3};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.83141},
                {0.60989},
                {0.5986}};
        b = new Matrix(bEntries);

        SparseCMatrix final1a = a;
        Matrix final1b = b;
        assertThrows(Exception.class, ()->final1a.setSlice(final1b, 0, 16));

        // ---------------------  Sub-case 6 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.02357+0.30556i"), new CNumber("0.85024+0.4537i"), new CNumber("0.7543+0.49265i"), new CNumber("0.75906+0.14116i")};
        aRowIndices = new int[]{0, 1, 1, 2};
        aColIndices = new int[]{3, 2, 3, 3};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.28434, 0.27412},
                {0.5756, 0.75022}};
        b = new Matrix(bEntries);

        SparseCMatrix final2a = a;
        Matrix final2b = b;
        assertThrows(Exception.class, ()->final2a.setSlice(final2b, 2, 0));

        // ---------------------  Sub-case 7 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.39067+0.57459i"), new CNumber("0.30883+0.44375i"), new CNumber("0.06197+0.10166i"), new CNumber("0.29584+0.81226i")};
        aRowIndices = new int[]{0, 0, 0, 1};
        aColIndices = new int[]{0, 2, 4, 3};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.41028, 0.03141, 0.92985}};
        b = new Matrix(bEntries);

        SparseCMatrix final3a = a;
        Matrix final3b = b;
        assertThrows(Exception.class, ()->final3a.setSlice(final3b, 0, 4));
    }


    @Test
    void setSliceDenseComplexArrayTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        SparseCMatrix a;

        CNumber[][] b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        SparseCMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.81854+0.73794i"), new CNumber("0.92149+0.91351i"), new CNumber("0.05864+0.50432i")};
        aRowIndices = new int[]{0, 1, 4};
        aColIndices = new int[]{2, 0, 2};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new CNumber[][]{
                {new CNumber("0.34615+0.93839i"), new CNumber("0.70113+0.08163i"), new CNumber("0.72114+0.24927i")},
                {new CNumber("0.39236+0.64693i"), new CNumber("0.08936+0.68308i"), new CNumber("0.46673+0.37432i")}};

        expShape = new Shape(5, 3);
        expEntries = new CNumber[]{new CNumber("0.81854+0.73794i"), new CNumber("0.92149+0.91351i"), new CNumber("0.34615+0.93839i"), new CNumber("0.70113+0.08163i"), new CNumber("0.72114+0.24927i"), new CNumber("0.39236+0.64693i"), new CNumber("0.08936+0.68308i"), new CNumber("0.46673+0.37432i"), new CNumber("0.05864+0.50432i")};
        expRowIndices = new int[]{0, 1, 2, 2, 2, 3, 3, 3, 4};
        expColIndices = new int[]{2, 0, 0, 1, 2, 0, 1, 2, 2};
        exp = new SparseCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setSlice(b, 2, 0));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(23, 11);
        aEntries = new CNumber[]{new CNumber("0.2977+0.49815i"), new CNumber("0.93231+0.3579i"), new CNumber("0.37551+0.5405i"), new CNumber("0.35116+0.64327i"), new CNumber("0.92754+0.50623i")};
        aRowIndices = new int[]{7, 10, 12, 16, 21};
        aColIndices = new int[]{2, 5, 3, 10, 7};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new CNumber[][]{
                {new CNumber("0.12905+0.39313i"), new CNumber("0.61059+0.52629i"), new CNumber("0.50152+0.83278i"), new CNumber("0.6374+0.66425i"), new CNumber("0.32806+0.78336i"), new CNumber("0.64419+0.28612i"), new CNumber("0.53083+0.66807i"), new CNumber("0.94756+0.86249i"), new CNumber("0.76154+0.8786i")},
                {new CNumber("0.49092+0.6522i"), new CNumber("0.32753+0.42391i"), new CNumber("0.41538+0.50721i"), new CNumber("0.13421+0.16425i"), new CNumber("0.58236+0.95564i"), new CNumber("0.59983+0.78352i"), new CNumber("0.18379+0.90525i"), new CNumber("0.28636+0.52966i"), new CNumber("0.4094+0.02304i")},
                {new CNumber("0.84913+0.26005i"), new CNumber("0.27942+0.2818i"), new CNumber("0.91324+0.80916i"), new CNumber("0.69438+0.99464i"), new CNumber("0.26807+0.68578i"), new CNumber("0.35027+0.18995i"), new CNumber("0.37641+0.26499i"), new CNumber("0.32544+0.30073i"), new CNumber("0.32972+0.12541i")},
                {new CNumber("0.93471+0.36426i"), new CNumber("0.15669+0.36544i"), new CNumber("0.38645+0.69536i"), new CNumber("0.15397+0.75875i"), new CNumber("0.60257+0.98316i"), new CNumber("0.20104+0.25877i"), new CNumber("0.82364+0.74003i"), new CNumber("0.03478+0.86282i"), new CNumber("0.74371+0.96392i")},
                {new CNumber("0.50878+0.85866i"), new CNumber("0.981+0.64309i"), new CNumber("0.65287+0.36897i"), new CNumber("0.70799+0.17862i"), new CNumber("0.82731+0.17388i"), new CNumber("0.25824+0.71081i"), new CNumber("0.9726+0.47426i"), new CNumber("0.89019+0.89163i"), new CNumber("0.2164+0.69192i")}};

        expShape = new Shape(23, 11);
        expEntries = new CNumber[]{new CNumber("0.2977+0.49815i"), new CNumber("0.93231+0.3579i"), new CNumber("0.37551+0.5405i"), new CNumber("0.35116+0.64327i"), new CNumber("0.12905+0.39313i"), new CNumber("0.61059+0.52629i"), new CNumber("0.50152+0.83278i"), new CNumber("0.6374+0.66425i"), new CNumber("0.32806+0.78336i"), new CNumber("0.64419+0.28612i"), new CNumber("0.53083+0.66807i"), new CNumber("0.94756+0.86249i"), new CNumber("0.76154+0.8786i"), new CNumber("0.49092+0.6522i"), new CNumber("0.32753+0.42391i"), new CNumber("0.41538+0.50721i"), new CNumber("0.13421+0.16425i"), new CNumber("0.58236+0.95564i"), new CNumber("0.59983+0.78352i"), new CNumber("0.18379+0.90525i"), new CNumber("0.28636+0.52966i"), new CNumber("0.4094+0.02304i"), new CNumber("0.84913+0.26005i"), new CNumber("0.27942+0.2818i"), new CNumber("0.91324+0.80916i"), new CNumber("0.69438+0.99464i"), new CNumber("0.26807+0.68578i"), new CNumber("0.35027+0.18995i"), new CNumber("0.37641+0.26499i"), new CNumber("0.32544+0.30073i"), new CNumber("0.32972+0.12541i"), new CNumber("0.93471+0.36426i"), new CNumber("0.15669+0.36544i"), new CNumber("0.38645+0.69536i"), new CNumber("0.15397+0.75875i"), new CNumber("0.60257+0.98316i"), new CNumber("0.20104+0.25877i"), new CNumber("0.82364+0.74003i"), new CNumber("0.03478+0.86282i"), new CNumber("0.74371+0.96392i"), new CNumber("0.50878+0.85866i"), new CNumber("0.981+0.64309i"), new CNumber("0.65287+0.36897i"), new CNumber("0.70799+0.17862i"), new CNumber("0.82731+0.17388i"), new CNumber("0.25824+0.71081i"), new CNumber("0.9726+0.47426i"), new CNumber("0.89019+0.89163i"), new CNumber("0.2164+0.69192i")};
        expRowIndices = new int[]{7, 10, 12, 16, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22};
        expColIndices = new int[]{2, 5, 3, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        exp = new SparseCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setSlice(b, 18, 1));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(1000, 5);
        aEntries = new CNumber[]{new CNumber("0.99587+0.77576i"), new CNumber("0.95599+0.98078i"), new CNumber("0.57285+0.48638i"), new CNumber("0.89326+0.41168i"), new CNumber("0.10412+0.51827i"), new CNumber("0.78999+0.21893i"), new CNumber("0.12036+0.22949i"), new CNumber("0.57747+0.46717i"), new CNumber("0.36522+0.47117i")};
        aRowIndices = new int[]{221, 307, 418, 525, 544, 849, 874, 964, 965};
        aColIndices = new int[]{1, 2, 1, 3, 4, 4, 4, 1, 2};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new CNumber[][]{
                {new CNumber("0.24174+0.77322i"), new CNumber("0.78613+0.01267i")},
                {new CNumber("0.22263+0.24569i"), new CNumber("0.69396+0.91033i")},
                {new CNumber("0.58864+0.45489i"), new CNumber("0.21105+0.71781i")}};

        expShape = new Shape(1000, 5);
        expEntries = new CNumber[]{new CNumber("0.24174+0.77322i"), new CNumber("0.78613+0.01267i"), new CNumber("0.22263+0.24569i"), new CNumber("0.69396+0.91033i"), new CNumber("0.58864+0.45489i"), new CNumber("0.21105+0.71781i"), new CNumber("0.99587+0.77576i"), new CNumber("0.95599+0.98078i"), new CNumber("0.57285+0.48638i"), new CNumber("0.89326+0.41168i"), new CNumber("0.10412+0.51827i"), new CNumber("0.78999+0.21893i"), new CNumber("0.12036+0.22949i"), new CNumber("0.57747+0.46717i"), new CNumber("0.36522+0.47117i")};
        expRowIndices = new int[]{0, 0, 1, 1, 2, 2, 221, 307, 418, 525, 544, 849, 874, 964, 965};
        expColIndices = new int[]{0, 1, 0, 1, 0, 1, 1, 2, 1, 3, 4, 4, 4, 1, 2};
        exp = new SparseCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setSlice(b, 0, 0));

        // ---------------------  Sub-case 4 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.6847+0.52621i"), new CNumber("0.7663+0.79176i"), new CNumber("0.70588+0.12213i"), new CNumber("0.91614+0.20048i")};
        aRowIndices = new int[]{0, 1, 1, 2};
        aColIndices = new int[]{4, 0, 1, 3};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new CNumber[][]{
                {new CNumber("0.94256+0.51916i"), new CNumber("0.63801+0.09854i"), new CNumber("0.43124+0.47178i")}};

        SparseCMatrix final0a = a;
        CNumber[][] final0b = b;
        assertThrows(Exception.class, ()->final0a.setSlice(final0b, -1, 2));

        // ---------------------  Sub-case 5 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.14947+0.59665i"), new CNumber("0.49504+0.14386i"), new CNumber("0.13119+0.15161i"), new CNumber("0.6447+0.54342i")};
        aRowIndices = new int[]{0, 1, 1, 2};
        aColIndices = new int[]{0, 1, 2, 2};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new CNumber[][]{
                {new CNumber("0.20784+0.48191i")},
                {new CNumber("0.76272+0.44928i")},
                {new CNumber("0.63682+0.9628i")}};

        SparseCMatrix final1a = a;
        CNumber[][] final1b = b;
        assertThrows(Exception.class, ()->final1a.setSlice(final1b, 0, 16));

        // ---------------------  Sub-case 6 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.13646+0.62927i"), new CNumber("0.80805+0.83393i"), new CNumber("0.97974+0.93209i"), new CNumber("0.54243+0.03783i")};
        aRowIndices = new int[]{1, 1, 2, 2};
        aColIndices = new int[]{3, 4, 0, 4};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new CNumber[][]{
                {new CNumber("0.79726+0.69333i"), new CNumber("0.9287+0.23619i")},
                {new CNumber("0.89293+0.45897i"), new CNumber("0.61354+0.52279i")}};

        SparseCMatrix final2a = a;
        CNumber[][] final2b = b;
        assertThrows(Exception.class, ()->final2a.setSlice(final2b, 2, 0));

        // ---------------------  Sub-case 7 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.31562+0.78864i"), new CNumber("0.17566+0.71583i"), new CNumber("0.93925+0.71462i"), new CNumber("0.04243+0.76878i")};
        aRowIndices = new int[]{1, 1, 2, 2};
        aColIndices = new int[]{3, 4, 1, 2};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new CNumber[][]{
                {new CNumber("0.05611+0.91305i"), new CNumber("0.62833+0.09243i"), new CNumber("0.02546+0.70173i")}};

        SparseCMatrix final3a = a;
        CNumber[][] final3b = b;
        assertThrows(Exception.class, ()->final3a.setSlice(final3b, 0, 4));
    }


    @Test
    void setSliceRealDenseArrayTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        SparseCMatrix a;

        double[][] b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        SparseCMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.92932+0.83391i"), new CNumber("0.75564+0.4746i"), new CNumber("0.67667+0.02131i")};
        aRowIndices = new int[]{1, 2, 3};
        aColIndices = new int[]{2, 1, 0};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new double[][]{
                {0.20641, 0.00627, 0.74659},
                {0.32765, 0.21592, 0.62641}};

        expShape = new Shape(5, 3);
        expEntries = new CNumber[]{new CNumber("0.92932+0.83391i"), new CNumber("0.20641"), new CNumber("0.00627"), new CNumber("0.74659"), new CNumber("0.32765"), new CNumber("0.21592"), new CNumber("0.62641")};
        expRowIndices = new int[]{1, 2, 2, 2, 3, 3, 3};
        expColIndices = new int[]{2, 0, 1, 2, 0, 1, 2};
        exp = new SparseCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setSlice(b, 2, 0));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(23, 11);
        aEntries = new CNumber[]{new CNumber("0.78261+0.81287i"), new CNumber("0.92047+0.1968i"), new CNumber("0.19346+0.66988i"), new CNumber("0.79317+0.5482i"), new CNumber("0.86173+0.26806i")};
        aRowIndices = new int[]{2, 12, 13, 20, 22};
        aColIndices = new int[]{8, 8, 8, 9, 3};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new double[][]{
                {0.42509, 0.86635, 0.49669, 0.12935, 0.58896, 0.55769, 0.74153, 0.3902, 0.30427},
                {0.47135, 0.27404, 0.77679, 0.18645, 0.14262, 0.34505, 0.96778, 0.08602, 0.04046},
                {0.17124, 0.48106, 0.81049, 0.98155, 0.92245, 0.22864, 0.24145, 0.44011, 0.50031},
                {0.73361, 0.23513, 0.61501, 0.95234, 0.48264, 0.74098, 0.69689, 0.41605, 0.41617},
                {0.58739, 0.71357, 0.91358, 0.35843, 0.94982, 0.39818, 0.24228, 0.26308, 0.30217}};

        expShape = new Shape(23, 11);
        expEntries = new CNumber[]{new CNumber("0.78261+0.81287i"), new CNumber("0.92047+0.1968i"), new CNumber("0.19346+0.66988i"), new CNumber("0.42509"), new CNumber("0.86635"), new CNumber("0.49669"), new CNumber("0.12935"), new CNumber("0.58896"), new CNumber("0.55769"), new CNumber("0.74153"), new CNumber("0.3902"), new CNumber("0.30427"), new CNumber("0.47135"), new CNumber("0.27404"), new CNumber("0.77679"), new CNumber("0.18645"), new CNumber("0.14262"), new CNumber("0.34505"), new CNumber("0.96778"), new CNumber("0.08602"), new CNumber("0.04046"), new CNumber("0.17124"), new CNumber("0.48106"), new CNumber("0.81049"), new CNumber("0.98155"), new CNumber("0.92245"), new CNumber("0.22864"), new CNumber("0.24145"), new CNumber("0.44011"), new CNumber("0.50031"), new CNumber("0.73361"), new CNumber("0.23513"), new CNumber("0.61501"), new CNumber("0.95234"), new CNumber("0.48264"), new CNumber("0.74098"), new CNumber("0.69689"), new CNumber("0.41605"), new CNumber("0.41617"), new CNumber("0.58739"), new CNumber("0.71357"), new CNumber("0.91358"), new CNumber("0.35843"), new CNumber("0.94982"), new CNumber("0.39818"), new CNumber("0.24228"), new CNumber("0.26308"), new CNumber("0.30217")};
        expRowIndices = new int[]{2, 12, 13, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22};
        expColIndices = new int[]{8, 8, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        exp = new SparseCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setSlice(b, 18, 1));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(1000, 5);
        aEntries = new CNumber[]{new CNumber("0.47086+0.22829i"), new CNumber("0.3198+0.27911i"), new CNumber("0.15627+0.18005i"), new CNumber("0.8154+0.86608i"), new CNumber("0.29655+0.07095i"), new CNumber("0.82134+0.18228i"), new CNumber("0.35707+0.25355i"), new CNumber("0.66031+0.70124i"), new CNumber("0.97115+0.2385i")};
        aRowIndices = new int[]{104, 108, 134, 162, 201, 243, 356, 599, 750};
        aColIndices = new int[]{1, 3, 1, 4, 0, 4, 1, 3, 3};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new double[][]{
                {0.37385, 0.16278},
                {0.16053, 0.78368},
                {0.81585, 0.09195}};

        expShape = new Shape(1000, 5);
        expEntries = new CNumber[]{new CNumber("0.37385"), new CNumber("0.16278"), new CNumber("0.16053"), new CNumber("0.78368"), new CNumber("0.81585"), new CNumber("0.09195"), new CNumber("0.47086+0.22829i"), new CNumber("0.3198+0.27911i"), new CNumber("0.15627+0.18005i"), new CNumber("0.8154+0.86608i"), new CNumber("0.29655+0.07095i"), new CNumber("0.82134+0.18228i"), new CNumber("0.35707+0.25355i"), new CNumber("0.66031+0.70124i"), new CNumber("0.97115+0.2385i")};
        expRowIndices = new int[]{0, 0, 1, 1, 2, 2, 104, 108, 134, 162, 201, 243, 356, 599, 750};
        expColIndices = new int[]{0, 1, 0, 1, 0, 1, 1, 3, 1, 4, 0, 4, 1, 3, 3};
        exp = new SparseCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setSlice(b, 0, 0));

        // ---------------------  Sub-case 4 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.76161+0.32703i"), new CNumber("0.86463+0.73067i"), new CNumber("0.65177+0.53146i"), new CNumber("0.23319+0.56897i")};
        aRowIndices = new int[]{1, 2, 2, 2};
        aColIndices = new int[]{0, 1, 2, 4};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new double[][]{
                {0.2427, 0.27463, 0.00804}};

        SparseCMatrix final0a = a;
        double[][] final0b = b;
        assertThrows(Exception.class, ()->final0a.setSlice(final0b, -1, 2));

        // ---------------------  Sub-case 5 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.41438+0.13745i"), new CNumber("0.62856+0.69366i"), new CNumber("0.40666+0.4999i"), new CNumber("0.81274+0.10541i")};
        aRowIndices = new int[]{0, 0, 1, 2};
        aColIndices = new int[]{1, 4, 3, 3};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new double[][]{
                {0.83141},
                {0.60989},
                {0.5986}};

        SparseCMatrix final1a = a;
        double[][] final1b = b;
        assertThrows(Exception.class, ()->final1a.setSlice(final1b, 0, 16));

        // ---------------------  Sub-case 6 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.02357+0.30556i"), new CNumber("0.85024+0.4537i"), new CNumber("0.7543+0.49265i"), new CNumber("0.75906+0.14116i")};
        aRowIndices = new int[]{0, 1, 1, 2};
        aColIndices = new int[]{3, 2, 3, 3};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new double[][]{
                {0.28434, 0.27412},
                {0.5756, 0.75022}};

        SparseCMatrix final2a = a;
        double[][] final2b = b;
        assertThrows(Exception.class, ()->final2a.setSlice(final2b, 2, 0));

        // ---------------------  Sub-case 7 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.39067+0.57459i"), new CNumber("0.30883+0.44375i"), new CNumber("0.06197+0.10166i"), new CNumber("0.29584+0.81226i")};
        aRowIndices = new int[]{0, 0, 0, 1};
        aColIndices = new int[]{0, 2, 4, 3};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new double[][]{
                {0.41028, 0.03141, 0.92985}};

        SparseCMatrix final3a = a;
        double[][] final3b = b;
        assertThrows(Exception.class, ()->final3a.setSlice(final3b, 0, 4));
    }


    @Test
    void setSliceRealDenseBoxedArrayTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        SparseCMatrix a;

        Double[][] b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        SparseCMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.92932+0.83391i"), new CNumber("0.75564+0.4746i"), new CNumber("0.67667+0.02131i")};
        aRowIndices = new int[]{1, 2, 3};
        aColIndices = new int[]{2, 1, 0};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new Double[][]{
                {0.20641, 0.00627, 0.74659},
                {0.32765, 0.21592, 0.62641}};

        expShape = new Shape(5, 3);
        expEntries = new CNumber[]{new CNumber("0.92932+0.83391i"), new CNumber("0.20641"), new CNumber("0.00627"), new CNumber("0.74659"), new CNumber("0.32765"), new CNumber("0.21592"), new CNumber("0.62641")};
        expRowIndices = new int[]{1, 2, 2, 2, 3, 3, 3};
        expColIndices = new int[]{2, 0, 1, 2, 0, 1, 2};
        exp = new SparseCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setSlice(b, 2, 0));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(23, 11);
        aEntries = new CNumber[]{new CNumber("0.78261+0.81287i"), new CNumber("0.92047+0.1968i"), new CNumber("0.19346+0.66988i"), new CNumber("0.79317+0.5482i"), new CNumber("0.86173+0.26806i")};
        aRowIndices = new int[]{2, 12, 13, 20, 22};
        aColIndices = new int[]{8, 8, 8, 9, 3};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new Double[][]{
                {0.42509, 0.86635, 0.49669, 0.12935, 0.58896, 0.55769, 0.74153, 0.3902, 0.30427},
                {0.47135, 0.27404, 0.77679, 0.18645, 0.14262, 0.34505, 0.96778, 0.08602, 0.04046},
                {0.17124, 0.48106, 0.81049, 0.98155, 0.92245, 0.22864, 0.24145, 0.44011, 0.50031},
                {0.73361, 0.23513, 0.61501, 0.95234, 0.48264, 0.74098, 0.69689, 0.41605, 0.41617},
                {0.58739, 0.71357, 0.91358, 0.35843, 0.94982, 0.39818, 0.24228, 0.26308, 0.30217}};

        expShape = new Shape(23, 11);
        expEntries = new CNumber[]{new CNumber("0.78261+0.81287i"), new CNumber("0.92047+0.1968i"), new CNumber("0.19346+0.66988i"), new CNumber("0.42509"), new CNumber("0.86635"), new CNumber("0.49669"), new CNumber("0.12935"), new CNumber("0.58896"), new CNumber("0.55769"), new CNumber("0.74153"), new CNumber("0.3902"), new CNumber("0.30427"), new CNumber("0.47135"), new CNumber("0.27404"), new CNumber("0.77679"), new CNumber("0.18645"), new CNumber("0.14262"), new CNumber("0.34505"), new CNumber("0.96778"), new CNumber("0.08602"), new CNumber("0.04046"), new CNumber("0.17124"), new CNumber("0.48106"), new CNumber("0.81049"), new CNumber("0.98155"), new CNumber("0.92245"), new CNumber("0.22864"), new CNumber("0.24145"), new CNumber("0.44011"), new CNumber("0.50031"), new CNumber("0.73361"), new CNumber("0.23513"), new CNumber("0.61501"), new CNumber("0.95234"), new CNumber("0.48264"), new CNumber("0.74098"), new CNumber("0.69689"), new CNumber("0.41605"), new CNumber("0.41617"), new CNumber("0.58739"), new CNumber("0.71357"), new CNumber("0.91358"), new CNumber("0.35843"), new CNumber("0.94982"), new CNumber("0.39818"), new CNumber("0.24228"), new CNumber("0.26308"), new CNumber("0.30217")};
        expRowIndices = new int[]{2, 12, 13, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22};
        expColIndices = new int[]{8, 8, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        exp = new SparseCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setSlice(b, 18, 1));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(1000, 5);
        aEntries = new CNumber[]{new CNumber("0.47086+0.22829i"), new CNumber("0.3198+0.27911i"), new CNumber("0.15627+0.18005i"), new CNumber("0.8154+0.86608i"), new CNumber("0.29655+0.07095i"), new CNumber("0.82134+0.18228i"), new CNumber("0.35707+0.25355i"), new CNumber("0.66031+0.70124i"), new CNumber("0.97115+0.2385i")};
        aRowIndices = new int[]{104, 108, 134, 162, 201, 243, 356, 599, 750};
        aColIndices = new int[]{1, 3, 1, 4, 0, 4, 1, 3, 3};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new Double[][]{
                {0.37385, 0.16278},
                {0.16053, 0.78368},
                {0.81585, 0.09195}};

        expShape = new Shape(1000, 5);
        expEntries = new CNumber[]{new CNumber("0.37385"), new CNumber("0.16278"), new CNumber("0.16053"), new CNumber("0.78368"), new CNumber("0.81585"), new CNumber("0.09195"), new CNumber("0.47086+0.22829i"), new CNumber("0.3198+0.27911i"), new CNumber("0.15627+0.18005i"), new CNumber("0.8154+0.86608i"), new CNumber("0.29655+0.07095i"), new CNumber("0.82134+0.18228i"), new CNumber("0.35707+0.25355i"), new CNumber("0.66031+0.70124i"), new CNumber("0.97115+0.2385i")};
        expRowIndices = new int[]{0, 0, 1, 1, 2, 2, 104, 108, 134, 162, 201, 243, 356, 599, 750};
        expColIndices = new int[]{0, 1, 0, 1, 0, 1, 1, 3, 1, 4, 0, 4, 1, 3, 3};
        exp = new SparseCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setSlice(b, 0, 0));

        // ---------------------  Sub-case 4 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.76161+0.32703i"), new CNumber("0.86463+0.73067i"), new CNumber("0.65177+0.53146i"), new CNumber("0.23319+0.56897i")};
        aRowIndices = new int[]{1, 2, 2, 2};
        aColIndices = new int[]{0, 1, 2, 4};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new Double[][]{
                {0.2427, 0.27463, 0.00804}};

        SparseCMatrix final0a = a;
        Double[][] final0b = b;
        assertThrows(Exception.class, ()->final0a.setSlice(final0b, -1, 2));

        // ---------------------  Sub-case 5 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.41438+0.13745i"), new CNumber("0.62856+0.69366i"), new CNumber("0.40666+0.4999i"), new CNumber("0.81274+0.10541i")};
        aRowIndices = new int[]{0, 0, 1, 2};
        aColIndices = new int[]{1, 4, 3, 3};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new Double[][]{
                {0.83141},
                {0.60989},
                {0.5986}};

        SparseCMatrix final1a = a;
        Double[][] final1b = b;
        assertThrows(Exception.class, ()->final1a.setSlice(final1b, 0, 16));

        // ---------------------  Sub-case 6 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.02357+0.30556i"), new CNumber("0.85024+0.4537i"), new CNumber("0.7543+0.49265i"), new CNumber("0.75906+0.14116i")};
        aRowIndices = new int[]{0, 1, 1, 2};
        aColIndices = new int[]{3, 2, 3, 3};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new Double[][]{
                {0.28434, 0.27412},
                {0.5756, 0.75022}};

        SparseCMatrix final2a = a;
        Double[][] final2b = b;
        assertThrows(Exception.class, ()->final2a.setSlice(final2b, 2, 0));

        // ---------------------  Sub-case 7 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.39067+0.57459i"), new CNumber("0.30883+0.44375i"), new CNumber("0.06197+0.10166i"), new CNumber("0.29584+0.81226i")};
        aRowIndices = new int[]{0, 0, 0, 1};
        aColIndices = new int[]{0, 2, 4, 3};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new Double[][]{
                {0.41028, 0.03141, 0.92985}};

        SparseCMatrix final3a = a;
        Double[][] final3b = b;
        assertThrows(Exception.class, ()->final3a.setSlice(final3b, 0, 4));
    }


    @Test
    void setSliceRealDenseIntBoxedArrayTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        SparseCMatrix a;

        Integer[][] b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        SparseCMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.92932+0.83391i"), new CNumber("0.75564+0.4746i"), new CNumber("0.67667+0.02131i")};
        aRowIndices = new int[]{1, 2, 3};
        aColIndices = new int[]{2, 1, 0};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new Integer[][]{
                {20641, 627, 74659},
                {32765, 21592, 62641}};

        expShape = new Shape(5, 3);
        expEntries = new CNumber[]{new CNumber("0.92932+0.83391i"), new CNumber("20641"), new CNumber("627"), new CNumber("74659"), new CNumber("32765"), new CNumber("21592"), new CNumber("62641")};
        expRowIndices = new int[]{1, 2, 2, 2, 3, 3, 3};
        expColIndices = new int[]{2, 0, 1, 2, 0, 1, 2};
        exp = new SparseCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setSlice(b, 2, 0));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(23, 11);
        aEntries = new CNumber[]{new CNumber("0.78261+0.81287i"), new CNumber("0.92047+0.1968i"), new CNumber("0.19346+0.66988i"), new CNumber("0.79317+0.5482i"), new CNumber("0.86173+0.26806i")};
        aRowIndices = new int[]{2, 12, 13, 20, 22};
        aColIndices = new int[]{8, 8, 8, 9, 3};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new Integer[][]{
                {42509, 86635, 49669, 12935, 58896, 55769, 74153, 3902, 30427},
                {47135, 27404, 77679, 18645, 14262, 34505, 96778, 8602, 4046},
                {17124, 48106, 81049, 98155, 92245, 22864, 24145, 44011, 50031},
                {73361, 23513, 61501, 95234, 48264, 74098, 69689, 41605, 41617},
                {58739, 71357, 91358, 35843, 94982, 39818, 24228, 26308, 30217}};

        expShape = new Shape(23, 11);
        expEntries = new CNumber[]{new CNumber("0.78261+0.81287i"), new CNumber("0.92047+0.1968i"), new CNumber("0.19346+0.66988i"), new CNumber("42509"), new CNumber("86635"), new CNumber("49669"), new CNumber("12935"), new CNumber("58896"), new CNumber("55769"), new CNumber("74153"), new CNumber("3902"), new CNumber("30427"), new CNumber("47135"), new CNumber("27404"), new CNumber("77679"), new CNumber("18645"), new CNumber("14262"), new CNumber("34505"), new CNumber("96778"), new CNumber("8602"), new CNumber("4046"), new CNumber("17124"), new CNumber("48106"), new CNumber("81049"), new CNumber("98155"), new CNumber("92245"), new CNumber("22864"), new CNumber("24145"), new CNumber("44011"), new CNumber("50031"), new CNumber("73361"), new CNumber("23513"), new CNumber("61501"), new CNumber("95234"), new CNumber("48264"), new CNumber("74098"), new CNumber("69689"), new CNumber("41605"), new CNumber("41617"), new CNumber("58739"), new CNumber("71357"), new CNumber("91358"), new CNumber("35843"), new CNumber("94982"), new CNumber("39818"), new CNumber("24228"), new CNumber("26308"), new CNumber("30217")};
        expRowIndices = new int[]{2, 12, 13, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22};
        expColIndices = new int[]{8, 8, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        exp = new SparseCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setSlice(b, 18, 1));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(1000, 5);
        aEntries = new CNumber[]{new CNumber("0.47086+0.22829i"), new CNumber("0.3198+0.27911i"), new CNumber("0.15627+0.18005i"), new CNumber("0.8154+0.86608i"), new CNumber("0.29655+0.07095i"), new CNumber("0.82134+0.18228i"), new CNumber("0.35707+0.25355i"), new CNumber("0.66031+0.70124i"), new CNumber("0.97115+0.2385i")};
        aRowIndices = new int[]{104, 108, 134, 162, 201, 243, 356, 599, 750};
        aColIndices = new int[]{1, 3, 1, 4, 0, 4, 1, 3, 3};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new Integer[][]{
                {37385, 16278},
                {16053, 78368},
                {81585, 9195}};

        expShape = new Shape(1000, 5);
        expEntries = new CNumber[]{new CNumber("37385"), new CNumber("16278"), new CNumber("16053"), new CNumber("78368"), new CNumber("81585"), new CNumber("9195"), new CNumber("0.47086+0.22829i"), new CNumber("0.3198+0.27911i"), new CNumber("0.15627+0.18005i"), new CNumber("0.8154+0.86608i"), new CNumber("0.29655+0.07095i"), new CNumber("0.82134+0.18228i"), new CNumber("0.35707+0.25355i"), new CNumber("0.66031+0.70124i"), new CNumber("0.97115+0.2385i")};
        expRowIndices = new int[]{0, 0, 1, 1, 2, 2, 104, 108, 134, 162, 201, 243, 356, 599, 750};
        expColIndices = new int[]{0, 1, 0, 1, 0, 1, 1, 3, 1, 4, 0, 4, 1, 3, 3};
        exp = new SparseCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setSlice(b, 0, 0));

        // ---------------------  Sub-case 4 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.76161+0.32703i"), new CNumber("0.86463+0.73067i"), new CNumber("0.65177+0.53146i"), new CNumber("0.23319+0.56897i")};
        aRowIndices = new int[]{1, 2, 2, 2};
        aColIndices = new int[]{0, 1, 2, 4};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new Integer[][]{
                {2427, 27463, 804}};

        SparseCMatrix final0a = a;
        Integer[][] final0b = b;
        assertThrows(Exception.class, ()->final0a.setSlice(final0b, -1, 2));

        // ---------------------  Sub-case 5 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.41438+0.13745i"), new CNumber("0.62856+0.69366i"), new CNumber("0.40666+0.4999i"), new CNumber("0.81274+0.10541i")};
        aRowIndices = new int[]{0, 0, 1, 2};
        aColIndices = new int[]{1, 4, 3, 3};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new Integer[][]{
                {83141},
                {60989},
                {5986}};

        SparseCMatrix final1a = a;
        Integer[][] final1b = b;
        assertThrows(Exception.class, ()->final1a.setSlice(final1b, 0, 16));

        // ---------------------  Sub-case 6 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.02357+0.30556i"), new CNumber("0.85024+0.4537i"), new CNumber("0.7543+0.49265i"), new CNumber("0.75906+0.14116i")};
        aRowIndices = new int[]{0, 1, 1, 2};
        aColIndices = new int[]{3, 2, 3, 3};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new Integer[][]{
                {28434, 27412},
                {5756, 75022}};

        SparseCMatrix final2a = a;
        Integer[][] final2b = b;
        assertThrows(Exception.class, ()->final2a.setSlice(final2b, 2, 0));

        // ---------------------  Sub-case 7 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.39067+0.57459i"), new CNumber("0.30883+0.44375i"), new CNumber("0.06197+0.10166i"), new CNumber("0.29584+0.81226i")};
        aRowIndices = new int[]{0, 0, 0, 1};
        aColIndices = new int[]{0, 2, 4, 3};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new Integer[][]{
                {41028, 3141, 92985}};

        SparseCMatrix final3a = a;
        Integer[][] final3b = b;
        assertThrows(Exception.class, ()->final3a.setSlice(final3b, 0, 4));
    }


    @Test
    void setSliceRealDenseIntArrayTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        SparseCMatrix a;

        int[][] b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        SparseCMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.92932+0.83391i"), new CNumber("0.75564+0.4746i"), new CNumber("0.67667+0.02131i")};
        aRowIndices = new int[]{1, 2, 3};
        aColIndices = new int[]{2, 1, 0};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new int[][]{
                {20641, 627, 74659},
                {32765, 21592, 62641}};

        expShape = new Shape(5, 3);
        expEntries = new CNumber[]{new CNumber("0.92932+0.83391i"), new CNumber("20641"), new CNumber("627"), new CNumber("74659"), new CNumber("32765"), new CNumber("21592"), new CNumber("62641")};
        expRowIndices = new int[]{1, 2, 2, 2, 3, 3, 3};
        expColIndices = new int[]{2, 0, 1, 2, 0, 1, 2};
        exp = new SparseCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setSlice(b, 2, 0));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(23, 11);
        aEntries = new CNumber[]{new CNumber("0.78261+0.81287i"), new CNumber("0.92047+0.1968i"), new CNumber("0.19346+0.66988i"), new CNumber("0.79317+0.5482i"), new CNumber("0.86173+0.26806i")};
        aRowIndices = new int[]{2, 12, 13, 20, 22};
        aColIndices = new int[]{8, 8, 8, 9, 3};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new int[][]{
                {42509, 86635, 49669, 12935, 58896, 55769, 74153, 3902, 30427},
                {47135, 27404, 77679, 18645, 14262, 34505, 96778, 8602, 4046},
                {17124, 48106, 81049, 98155, 92245, 22864, 24145, 44011, 50031},
                {73361, 23513, 61501, 95234, 48264, 74098, 69689, 41605, 41617},
                {58739, 71357, 91358, 35843, 94982, 39818, 24228, 26308, 30217}};

        expShape = new Shape(23, 11);
        expEntries = new CNumber[]{new CNumber("0.78261+0.81287i"), new CNumber("0.92047+0.1968i"), new CNumber("0.19346+0.66988i"), new CNumber("42509"), new CNumber("86635"), new CNumber("49669"), new CNumber("12935"), new CNumber("58896"), new CNumber("55769"), new CNumber("74153"), new CNumber("3902"), new CNumber("30427"), new CNumber("47135"), new CNumber("27404"), new CNumber("77679"), new CNumber("18645"), new CNumber("14262"), new CNumber("34505"), new CNumber("96778"), new CNumber("8602"), new CNumber("4046"), new CNumber("17124"), new CNumber("48106"), new CNumber("81049"), new CNumber("98155"), new CNumber("92245"), new CNumber("22864"), new CNumber("24145"), new CNumber("44011"), new CNumber("50031"), new CNumber("73361"), new CNumber("23513"), new CNumber("61501"), new CNumber("95234"), new CNumber("48264"), new CNumber("74098"), new CNumber("69689"), new CNumber("41605"), new CNumber("41617"), new CNumber("58739"), new CNumber("71357"), new CNumber("91358"), new CNumber("35843"), new CNumber("94982"), new CNumber("39818"), new CNumber("24228"), new CNumber("26308"), new CNumber("30217")};
        expRowIndices = new int[]{2, 12, 13, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22};
        expColIndices = new int[]{8, 8, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        exp = new SparseCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setSlice(b, 18, 1));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(1000, 5);
        aEntries = new CNumber[]{new CNumber("0.47086+0.22829i"), new CNumber("0.3198+0.27911i"), new CNumber("0.15627+0.18005i"), new CNumber("0.8154+0.86608i"), new CNumber("0.29655+0.07095i"), new CNumber("0.82134+0.18228i"), new CNumber("0.35707+0.25355i"), new CNumber("0.66031+0.70124i"), new CNumber("0.97115+0.2385i")};
        aRowIndices = new int[]{104, 108, 134, 162, 201, 243, 356, 599, 750};
        aColIndices = new int[]{1, 3, 1, 4, 0, 4, 1, 3, 3};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new int[][]{
                {37385, 16278},
                {16053, 78368},
                {81585, 9195}};

        expShape = new Shape(1000, 5);
        expEntries = new CNumber[]{new CNumber("37385"), new CNumber("16278"), new CNumber("16053"), new CNumber("78368"), new CNumber("81585"), new CNumber("9195"), new CNumber("0.47086+0.22829i"), new CNumber("0.3198+0.27911i"), new CNumber("0.15627+0.18005i"), new CNumber("0.8154+0.86608i"), new CNumber("0.29655+0.07095i"), new CNumber("0.82134+0.18228i"), new CNumber("0.35707+0.25355i"), new CNumber("0.66031+0.70124i"), new CNumber("0.97115+0.2385i")};
        expRowIndices = new int[]{0, 0, 1, 1, 2, 2, 104, 108, 134, 162, 201, 243, 356, 599, 750};
        expColIndices = new int[]{0, 1, 0, 1, 0, 1, 1, 3, 1, 4, 0, 4, 1, 3, 3};
        exp = new SparseCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setSlice(b, 0, 0));

        // ---------------------  Sub-case 4 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.76161+0.32703i"), new CNumber("0.86463+0.73067i"), new CNumber("0.65177+0.53146i"), new CNumber("0.23319+0.56897i")};
        aRowIndices = new int[]{1, 2, 2, 2};
        aColIndices = new int[]{0, 1, 2, 4};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new int[][]{
                {2427, 27463, 804}};

        SparseCMatrix final0a = a;
        int[][] final0b = b;
        assertThrows(Exception.class, ()->final0a.setSlice(final0b, -1, 2));

        // ---------------------  Sub-case 5 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.41438+0.13745i"), new CNumber("0.62856+0.69366i"), new CNumber("0.40666+0.4999i"), new CNumber("0.81274+0.10541i")};
        aRowIndices = new int[]{0, 0, 1, 2};
        aColIndices = new int[]{1, 4, 3, 3};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new int[][]{
                {83141},
                {60989},
                {5986}};

        SparseCMatrix final1a = a;
        int[][] final1b = b;
        assertThrows(Exception.class, ()->final1a.setSlice(final1b, 0, 16));

        // ---------------------  Sub-case 6 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.02357+0.30556i"), new CNumber("0.85024+0.4537i"), new CNumber("0.7543+0.49265i"), new CNumber("0.75906+0.14116i")};
        aRowIndices = new int[]{0, 1, 1, 2};
        aColIndices = new int[]{3, 2, 3, 3};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new int[][]{
                {28434, 27412},
                {5756, 75022}};

        SparseCMatrix final2a = a;
        int[][] final2b = b;
        assertThrows(Exception.class, ()->final2a.setSlice(final2b, 2, 0));

        // ---------------------  Sub-case 7 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.39067+0.57459i"), new CNumber("0.30883+0.44375i"), new CNumber("0.06197+0.10166i"), new CNumber("0.29584+0.81226i")};
        aRowIndices = new int[]{0, 0, 0, 1};
        aColIndices = new int[]{0, 2, 4, 3};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        b = new int[][]{
                {41028, 3141, 92985}};

        SparseCMatrix final3a = a;
        int[][] final3b = b;
        assertThrows(Exception.class, ()->final3a.setSlice(final3b, 0, 4));
    }
}
