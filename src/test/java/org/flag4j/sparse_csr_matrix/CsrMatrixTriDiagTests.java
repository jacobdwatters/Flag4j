package org.flag4j.sparse_csr_matrix;

import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.flag4j.core.Shape;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CsrMatrixTriDiagTests {

    static CsrMatrix A;
    static Shape aShape;
    static double[] aEntries;
    static int[] aRowPointers;
    static int[] aColIndices;

    @Test
    void traceTest() {
        // ----------------------- sub-case 1 -----------------------
        aShape = new Shape(6, 6);
        aEntries = new double[]{0.550651808914819, 0.25248902903617765, 0.7347711759989852, 0.6355561756397741, 0.4229489382733195};
        aRowPointers = new int[]{0, 1, 3, 3, 4, 4, 5};
        aColIndices = new int[]{1, 0, 3, 4, 2};
        A = new CsrMatrix(aShape, aEntries, aRowPointers, aColIndices);
        assertEquals(0.0, A.trace());

        // ----------------------- sub-case 2 -----------------------
        aShape = new Shape(6, 6);
        aEntries = new double[]{0.3947350230967598, 0.057362465663433504, 0.9347345786066202, 0.43387856849809536, 0.644454913626892};
        aRowPointers = new int[]{0, 2, 2, 2, 3, 3, 5};
        aColIndices = new int[]{2, 5, 3, 0, 3};
        A = new CsrMatrix(aShape, aEntries, aRowPointers, aColIndices);
        assertEquals(0.9347345786066202, A.trace());

        // ----------------------- sub-case 3 -----------------------
        aShape = new Shape(15, 15);
        aEntries = new double[]{0.8086884822224377, 0.8263266272760342, 0.32525649651881317, 0.9608789667749497, 0.4395049603468911,
                0.07305897091345792, 0.7678491080726771, 0.2294487507606635, 0.1382399153004038, 0.920903220380491, 0.6895253484430522,
                0.9256640817501549, 0.8520473820197568, 0.6642547057142691, 0.9992544305209986, 0.6144764840301362,
                0.38299169954163803, 0.04890094352081187, 0.07595056410539092, 0.6778873127860436, 0.5725538366855386,
                0.633943368653018};
        aRowPointers = new int[]{0, 2, 5, 5, 6, 9, 10, 13, 14, 15, 15, 16, 17, 20, 21, 22};
        aColIndices = new int[]{8, 9, 1, 2, 13, 14, 0, 11, 13, 7, 8, 12, 14, 8, 14, 5, 4, 1, 4, 12, 9, 9};
        A = new CsrMatrix(aShape, aEntries, aRowPointers, aColIndices);
        assertEquals(1.0031438093048568, A.trace());

        // ----------------------- sub-case 4 -----------------------
        assertThrows(LinearAlgebraException.class, ()->new CsrMatrix(12, 4).tr());
    }


    @Test
    void getDiagTests() {
        CooVector exp;

        // ----------------------- sub-case 1 -----------------------
        aShape = new Shape(6, 6);
        aEntries = new double[]{0.2742696420912182, 0.07751690909235676, 0.921830928702064, 0.1935807720906103, 0.7540620527220376};
        aRowPointers = new int[]{0, 0, 3, 3, 4, 5, 5};
        aColIndices = new int[]{0, 2, 4, 5, 4};
        A = new CsrMatrix(aShape, aEntries, aRowPointers, aColIndices);
        exp = new Vector(0.0, 0.0, 0.0, 0.0, 0.7540620527220376, 0.0).toCoo();
        assertEquals(exp, A.getDiag());

        // ----------------------- sub-case 2 -----------------------
        aShape = new Shape(6, 6);
        aEntries = new double[]{0.5117934899539107, 0.5232442605382139, 0.6241020346691114, 0.5149866370978797, 0.9899310737230431, 0.18946387228386474, 0.2824144399493015, 0.25137905573116825, 0.9609185162323945, 0.6746428794940532, 0.47997193451501086};
        aRowPointers = new int[]{0, 2, 5, 7, 9, 10, 11};
        aColIndices = new int[]{2, 5, 0, 1, 3, 2, 5, 0, 4, 1, 5};
        A = new CsrMatrix(aShape, aEntries, aRowPointers, aColIndices);
        exp = new Vector(0.0, 0.5149866370978797, 0.18946387228386474, 0.0, 0.0, 0.47997193451501086).toCoo();
        assertEquals(exp, A.getDiag());

        // ----------------------- sub-case 3 -----------------------
        aShape = new Shape(12, 12);
        aEntries = new double[]{0.989336440083382, 0.5784226184530201, 0.4128615264146236, 0.8720518156451399, 0.015332897118263578, 0.30602701869807314, 0.5445047402785209, 0.6799626535731805, 0.8160519253929756, 0.44554133866387846, 0.5507111109163054, 0.5127437616463539, 0.12630665534801888, 0.8348974142473102};
        aRowPointers = new int[]{0, 1, 3, 5, 8, 9, 9, 10, 12, 12, 13, 14, 14};
        aColIndices = new int[]{6, 6, 8, 7, 10, 3, 6, 8, 4, 5, 3, 5, 3, 4};
        A = new CsrMatrix(aShape, aEntries, aRowPointers, aColIndices);
        exp = new Vector(0.0, 0.0, 0.0, 0.30602701869807314, 0.8160519253929756, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).toCoo();
        assertEquals(exp, A.getDiag());

        // ----------------------- sub-case 4 -----------------------
        aShape = new Shape(3, 7);
        aEntries = new double[]{0.46451579858863223, 0.761818146777147, 0.4513656938478876, 0.8045385713145264};
        aRowPointers = new int[]{0, 0, 2, 4};
        aColIndices = new int[]{1, 4, 1, 2};
        A = new CsrMatrix(aShape, aEntries, aRowPointers, aColIndices);
        exp = new Vector(0.0, 0.46451579858863223, 0.8045385713145264).toCoo();
        assertEquals(exp, A.getDiag());

        // ----------------------- sub-case 5 -----------------------
        aShape = new Shape(7, 3);
        aEntries = new double[]{0.11343649359086139, 0.6082353444109962, 0.8703305006626817, 0.9546036038577662};
        aRowPointers = new int[]{0, 0, 2, 3, 4, 4, 4, 4};
        aColIndices = new int[]{1, 2, 2, 0};
        A = new CsrMatrix(aShape, aEntries, aRowPointers, aColIndices);
        exp = new Vector(0.0, 0.11343649359086139, 0.8703305006626817).toCoo();
        assertEquals(exp, A.getDiag());
    }
}
