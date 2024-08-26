package org.flag4j.complex_matrix;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.sparse.CsrMatrixOld;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CMatrixCsrMatMultTests {
    CMatrixOld A;
    CNumber[][] aEntries;

    CsrMatrixOld B;
    Shape bShape;
    double[] bEntries;
    int[] bRowPointers;
    int[] bColIndices;

    CMatrixOld exp;
    CNumber[][] expEntries;


    @Test
    void standardTests() {
        // ------------------------ Sub-case 1 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber("0.1013+0.5667i"), new CNumber("0.56204+0.08795i"), new CNumber("0.26919+0.74589i"), new CNumber("0.40605+0.37181i"), new CNumber("0.41665+0.56373i")},
                {new CNumber("0.88631+0.70806i"), new CNumber("0.94873+0.70971i"), new CNumber("0.73508+0.92932i"), new CNumber("0.32551+0.08181i"), new CNumber("0.80165+0.87963i")},
                {new CNumber("0.01923+0.20639i"), new CNumber("0.01025+0.53356i"), new CNumber("0.77862+0.04428i"), new CNumber("0.24381+0.01189i"), new CNumber("0.5903+0.51795i")},
                {new CNumber("0.65994+0.40064i"), new CNumber("0.21257+0.16288i"), new CNumber("0.85927+0.11806i"), new CNumber("0.8716+0.70231i"), new CNumber("0.83819+0.21429i")},
                {new CNumber("0.29866+0.71364i"), new CNumber("0.46553+0.89626i"), new CNumber("0.25626+0.07154i"), new CNumber("0.27779+0.59077i"), new CNumber("0.3676+0.45885i")}};
        A = new CMatrixOld(aEntries);
        bShape = new Shape(5, 5);
        bEntries = new double[]{0.35336, 0.80623, 0.7923, 0.96503, 0.33155, 0.26233, 0.93305, 0.97519};
        bRowPointers = new int[]{0, 2, 3, 7, 8, 8};
        bColIndices = new int[]{1, 4, 4, 0, 1, 2, 3, 4};
        B = new CsrMatrixOld(bShape, bEntries, bRowPointers, bColIndices);
        expEntries = new CNumber[][]{
                {new CNumber("0.2597764257+0.7198062267i"), new CNumber("0.1250453125+0.4475489415i"), new CNumber("0.0706166127+0.1956693237i"), new CNumber("0.2511677295+0.6959526645i"), new CNumber("0.9229512905+0.8891587199i")},
                {new CNumber("0.7093742524+0.8968216796i"), new CNumber("0.5569022755999999+0.5583161276i"), new CNumber("0.1928335364+0.2437885156i"), new CNumber("0.685866394+0.8671020260000001i"), new CNumber("1.7836825872+1.2129427407000002i")},
                {new CNumber("0.7513916586+0.0427315284i"), new CNumber("0.26494657380000003+0.0876110044i"), new CNumber("0.2042553846+0.0116159724i"), new CNumber("0.726491391+0.041315454i"), new CNumber("0.2613859518+0.6007324068000001i")},
                {new CNumber("0.8292213281+0.1139314418i"), new CNumber("0.5180873669+0.18071294340000002i"), new CNumber("0.2254122991+0.0309706798i"), new CNumber("0.8017418735+0.11015588300000001i"), new CNumber("1.5504582412+1.1369435001000001i")},
                {new CNumber("0.2472985878+0.06903824620000001i"), new CNumber("0.19049750059999998+0.27589091740000005i"), new CNumber("0.0672246858+0.018767088200000004i"), new CNumber("0.239103393+0.066750397i"), new CNumber("0.8805261008999998+1.8615777715i")}};
        exp = new CMatrixOld(expEntries);
        assertEquals(exp, A.mult(B));

        // ------------------------ Sub-case 2 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber("0.66805+0.1356i"), new CNumber("0.02304+0.45698i"), new CNumber("0.21488+0.42425i")},
                {new CNumber("0.41083+0.05417i"), new CNumber("0.73696+0.55199i"), new CNumber("0.14116+0.82652i")},
                {new CNumber("0.37502+0.66468i"), new CNumber("0.0909+0.08818i"), new CNumber("0.67084+0.0016i")},
                {new CNumber("0.57517+0.93909i"), new CNumber("0.42799+0.32726i"), new CNumber("0.92712+0.44066i")},
                {new CNumber("0.9064+0.65531i"), new CNumber("0.12811+0.44938i"), new CNumber("0.3343+0.45097i")},
                {new CNumber("0.49565+0.04573i"), new CNumber("0.84561+0.56477i"), new CNumber("0.5758+0.99876i")},
                {new CNumber("0.82457+0.04468i"), new CNumber("0.73855+0.10166i"), new CNumber("0.23595+0.92517i")},
                {new CNumber("0.93964+0.6983i"), new CNumber("0.80439+0.29492i"), new CNumber("0.15538+0.7558i")},
                {new CNumber("0.17902+0.59831i"), new CNumber("0.78638+0.43736i"), new CNumber("0.33244+0.23329i")},
                {new CNumber("0.6007+0.16697i"), new CNumber("0.54194+0.05845i"), new CNumber("0.64488+0.21455i")},
                {new CNumber("0.26206+0.22261i"), new CNumber("0.4165+0.64287i"), new CNumber("0.27519+0.17844i")}};
        A = new CMatrixOld(aEntries);

        bShape = new Shape(3, 10);
        bEntries = new double[]{0.58579, 0.44167, 0.79592};
        bRowPointers = new int[]{0, 1, 3, 3};
        bColIndices = new int[]{1, 4, 5};
        B = new CsrMatrixOld(bShape, bEntries, bRowPointers, bColIndices);

        expEntries = new CNumber[][]{
                {new CNumber("0.0"), new CNumber("0.39133700950000005+0.07943312400000001i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.010176076800000001+0.20183435660000001i"), new CNumber("0.0183379968+0.3637195216i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.2406601057+0.031732244300000004i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.32549312319999996+0.24379742329999998i"), new CNumber("0.5865612031999999+0.43933988079999997i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.21968296580000002+0.38936289720000006i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.040147802999999996+0.038946460599999996i"), new CNumber("0.072349128+0.0701842256i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.3369288343+0.5501095311i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.1890303433+0.1445409242i"), new CNumber("0.34064580079999995+0.2604727792i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.5309600560000001+0.3838740449i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0565823437+0.1984776646i"), new CNumber("0.10196531119999999+0.3576705296i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.2903468135+0.026788176700000003i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.3734805687+0.2494419659i"), new CNumber("0.6730379111999999+0.44951173839999997i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.4830248603+0.0261730972i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.3261953785+0.0449001722i"), new CNumber("0.587826716+0.08091322719999999i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.5504317156+0.40905715700000006i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.35527493130000004+0.13025731640000002i"), new CNumber("0.6402300888+0.2347327264i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.10486812580000002+0.35048401490000003i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.3473204546+0.19316879120000002i"), new CNumber("0.6258955695999999+0.3481035712i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.35188405300000003+0.09780935630000001i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.2393586398+0.025815611500000002i"), new CNumber("0.43134088479999994+0.046521524i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.15351212740000003+0.1304027119i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.183955555+0.2839363929i"), new CNumber("0.33150068+0.5116730904i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.mult(B));

        // ------------------------ Sub-case 3 ------------------------
        A = new CMatrixOld(24, 516);
        B = new CsrMatrixOld(15, 12);
        assertThrows(LinearAlgebraException.class, ()->A.mult(B));
    }
}
