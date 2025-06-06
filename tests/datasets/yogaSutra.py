## contains the entire YogaSutra as a list of strings. 
## used to test the functions in the process_sanskrit module

ys = [
 'atha yogānuśāsanam ',
 'yogaś cittavṛttinirodhaḥ ',
 "tadā draṣṭuḥ svarūpe 'vasthānam ",
 'vṛttisārūpyam itaratra ',
 'vṛttayaḥ pañcatayyaḥ kliṣṭākliṣṭāḥ ',
 'pramāṇaviparyayavikalpanidrāsmṛtayaḥ ',
 'pratyakṣānumānāgamāḥ pramāṇāni ',
 'viparyayo mithyājñānam atadrūpapratiṣṭham ',
 'śabdajñānānupātī vastuśūnyo vikalpaḥ ',
 'abhāvapratyayālambanā vṛttir nidrā ',
 'anubhūtaviṣayāsaṃpramoṣaḥ smṛtiḥ ',
 'abhyāsavairāgyābhyāṃ tannirodhaḥ ',
 "tatra sthitau yatno 'bhyāsaḥ ",
 'sa tu dīrghakālanairantaryasatkārāsevito dṛḍhabhūmiḥ ',
 'dṛṣṭānuśravikaviṣayavitṛṣṇasya vaśīkārasaṃjñā vairāgyam ',
 'tat paraṃ puruṣakhyāter guṇavaitṛṣṇyam ',
 'vitarkavicārānandāsmitārūpānugamāt saṃprajñātaḥ ',
 "virāmapratyayābhyāsapūrvaḥ saṃskāraśeṣo 'nyaḥ ",
 'bhavapratyayo videhaprakṛtilayānām ',
 'śraddhāvīryasmṛtisamādhiprajñāpūrvaka itareṣām ',
 'tīvrasaṃvegānām āsannaḥ ',
 "mṛdumadhyādhimātratvāt tato 'pi viśeṣaḥ ",
 'īśvarapraṇidhānād vā ',
 'kleśakarmavipākāśayair aparāmṛṣṭaḥ puruṣaviśeṣa īśvaraḥ ',
 'tatra niratiśayaṃ sarvajñabījam ',
 'pūrveṣām api guruḥ kālenānavacchedāt ',
 'tasya vācakaḥ praṇavaḥ ',
 'tajjapas tadarthabhāvanam ',
 "tataḥ pratyakcetanādhigamo 'py antarāyābhāvaś ca ",
 "vyādhistyānasaṃśayapramādālasyāviratibhrāntidarśanālabdhabhūmikatvānavasthitatvāni cittavikṣepās te 'ntarāyāḥ ",
 'duḥkhadaurmanasyāṅgamejayatvaśvāsapraśvāsā vikṣepasahabhuvaḥ ',
 'tatpratiṣedhārtham ekatattvābhyāsaḥ ',
 'maitrīkaruṇāmuditopekṣāṇāṃ sukhaduḥkhapuṇyāpuṇyaviṣayāṇāṃ bhāvanātaś cittaprasādanam ',
 'pracchardanavidhāraṇābhyāṃ vā prāṇasya ',
 'viṣayavatī vā pravṛttir utpannā manasaḥ sthitinibandhanī ',
 'viśokā vā jyotiṣmatī ',
 'vītarāgaviṣayaṃ vā cittam ',
 'svapnanidrājñānālambanaṃ vā ',
 'yathābhimatadhyānād vā ',
 "paramāṇuparamamahattvānto 'sya vaśīkāraḥ ",
 'kṣīṇavṛtter abhijātasyeva maṇer grahītṛgrahaṇagrāhyeṣu tatsthatadañjanatā samāpattiḥ ',
 'tatra śabdārthajñānavikalpaiḥ saṃkīrṇā savitarkā samāpattiḥ ',
 'smṛtipariśuddhau svarūpaśūnyevārthamātranirbhāsā nirvitarkā ',
 'etayaiva savicārā nirvicārā ca sūkṣmaviṣayā vyākhyātā ',
 'sūkṣmaviṣayatvaṃ cāliṅgaparyavasānam ',
 'tā eva sabījaḥ samādhiḥ ',
 "nirvicāravaiśāradye 'dhyātmaprasādaḥ ",
 'ṛtaṃbharā tatra prajñā ',
 'śrutānumānaprajñābhyām anyaviṣayā viśeṣārthatvāt ',
 "tajjaḥ saṃskāro 'nyasaṃskārapratibandhī ",
 'tasyāpi nirodhe sarvanirodhān nirbījaḥ samādhiḥ ',
 'tapaḥsvādhyāyeśvarapraṇidhānāni kriyāyogaḥ ',
 'samādhibhāvanārthaḥ kleśatanūkaraṇārthaś ca ',
 'avidyāsmitārāgadveṣābhiniveśāḥ kleśāḥ ',
 'avidyā kṣetram uttareṣāṃ prasuptatanuvicchinnodārāṇām ',
 'anityāśuciduḥkhānātmasu nityaśucisukhātmakhyātir avidyā ',
 'dṛgdarśanaśaktyor ekātmatevāsmitā ',
 'sukhānuśayī rāgaḥ ',
 'duḥkhānuśayī dveṣaḥ ',
 "svarasavāhī viduṣo 'pi tathā rūḍho 'bhiniveśaḥ ",
 'te pratiprasavaheyāḥ sūkṣmāḥ ',
 'dhyānaheyās tadvṛttayaḥ ',
 'kleśamūlaḥ karmāśayo dṛṣṭādṛṣṭajanmavedanīyaḥ ',
 'sati mūle tadvipāko jātyāyurbhogāḥ ',
 'te hlādaparitāpaphalāḥ puṇyāpuṇyahetutvāt ',
 'pariṇāmatāpasaṃskāraduḥkhair guṇavṛttivirodhāc ca duḥkham eva sarvaṃ vivekinaḥ ',
 'heyaṃ duḥkham anāgatam ',
 'draṣṭṛdṛśyayoḥ saṃyogo heyahetuḥ ',
 'prakāśakriyāsthitiśīlaṃ bhūtendriyātmakaṃ bhogāpavargārthaṃ dṛśyam ',
 'viśeṣāviśeṣaliṅgamātrāliṅgāni guṇaparvāṇi ',
 "draṣṭā dṛśimātraḥ śuddho 'pi pratyayānupaśyaḥ ",
 'tadartha eva dṛśyasyātmā ',
 'kṛtārthaṃ prati naṣṭam apy anaṣṭaṃ tadanyasādhāraṇatvāt ',
 'svasvāmiśaktyoḥ svarūpopalabdhihetuḥ saṃyogaḥ ',
 'tasya hetur avidyā ',
 'tadabhāvāt saṃyogābhāvo hānaṃ taddṛśeḥ kaivalyam ',
 'vivekakhyātir aviplavā hānopāyaḥ ',
 'tasya saptadhā prāntabhūmiḥ prajñā ',
 'yogāṅgānuṣṭhānād aśuddhikṣaye jñānadīptir ā vivekakhyāteḥ ',
 "yamaniyamāsanaprāṇāyāmapratyāhāradhāraṇādhyānasamādhayo 'ṣṭāv aṅgāni ",
 'ahiṃsāsatyāsteyabrahmacaryāparigrahā yamāḥ ',
 'jātideśakālasamayānavacchinnāḥ sārvabhaumā mahāvratam ',
 'śaucasaṃtoṣatapaḥsvādhyāyeśvarapraṇidhānāni niyamāḥ ',
 'vitarkabādhane pratipakṣabhāvanam ',
 'vitarkā hiṃsādayaḥ kṛtakāritānumoditā lobhakrodhamohapūrvakā mṛdumadhyādhimātrā duḥkhājñānānantaphalā iti pratipakṣabhāvanam ',
 'ahiṃsāpratiṣṭhāyāṃ tatsaṃnidhau vairatyāgaḥ ',
 'satyapratiṣṭhāyāṃ kriyāphalāśrayatvam ',
 'asteyapratiṣṭhāyāṃ sarvaratnopasthānam ',
 'brahmacaryapratiṣṭhāyāṃ vīryalābhaḥ ',
 'aparigrahasthairye janmakathaṃtāsaṃbodhaḥ ',
 'śaucāt svāṅgajugupsā parair asaṃsargaḥ ',
 'sattvaśuddhisaumanasyaikāgryendriyajayātmadarśanayogyatvāni ca ',
 'saṃtoṣād anuttamaḥ sukhalābhaḥ ',
 'kāyendriyasiddhir aśuddhikṣayāt tapasaḥ ',
 'svādhyāyād iṣṭadevatāsaṃprayogaḥ ',
 'samādhisiddhir īśvarapraṇidhānāt ',
 'sthirasukham āsanam ',
 'prayatnaśaithilyānantasamāpattibhyām ',
 'tato dvandvānabhighātaḥ ',
 'tasmin sati śvāsapraśvāsayor gativicchedaḥ prāṇāyāmaḥ ',
 'bāhyābhyantarastambhavṛttir deśakālasaṃkhyābhiḥ paridṛṣṭo dīrghasūkṣmaḥ ',
 'bāhyābhyantaraviṣayākṣepī caturthaḥ ',
 'tataḥ kṣīyate prakāśāvaraṇam ',
 'dhāraṇāsu ca yogyatā manasaḥ ',
 'svaviṣayāsaṃprayoge cittasvarūpānukāra ivendriyāṇāṃ pratyāhāraḥ ',
 'tataḥ paramā vaśyatendriyāṇām ',
 'deśabandhaś cittasya dhāraṇā ',
 'tatra pratyayaikatānatā dhyānam ',
 'tad evārthamātranirbhāsaṃ svarūpaśūnyam iva samādhiḥ ',
 'trayam ekatra saṃyamaḥ ',
 'tajjayāt prajñālokaḥ ',
 'tasya bhūmiṣu viniyogaḥ ',
 'trayam antaraṅgaṃ pūrvebhyaḥ ',
 'tad api bahiraṅgaṃ nirbījasya ',
 'vyutthānanirodhasaṃskārayor abhibhavaprādurbhāvau nirodhakṣaṇacittānvayo nirodhapariṇāmaḥ ',
 'tasya praśāntavāhitā saṃskārāt ',
 'sarvārthataikāgratayoḥ kṣayodayau cittasya samādhipariṇāmaḥ ',
 'tataḥ punaḥ śāntoditau tulyapratyayau cittasyaikāgratāpariṇāmaḥ ',
 'etena bhūtendriyeṣu dharmalakṣaṇāvasthāpariṇāmā vyākhyātāḥ ',
 'śāntoditāvyapadeśyadharmānupātī dharmī ',
 'kramānyatvaṃ pariṇāmānyatve hetuḥ ',
 'pariṇāmatrayasaṃyamād atītānāgatajñānam ',
 'śabdārthapratyayānām itaretarādhyāsāt saṃkaras tatpravibhāgasaṃyamāt sarvabhūtarutajñānam ',
 'saṃskārasākṣātkaraṇāt pūrvajātijñānam ',
 'pratyayasya paracittajñānam ',
 'na ca tat sālambanaṃ tasyāviṣayībhūtatvāt ',
 "kāyarūpasaṃyamāt tadgrāhyaśaktistambhe cakṣuḥprakāśāsaṃprayoge 'ntardhānam ",
 'sopakramaṃ nirupakramaṃ ca karma tatsaṃyamād aparāntajñānam ariṣṭebhyo vā ',
 'maitryādiṣu balāni ',
 'baleṣu hastibalādīni ',
 'pravṛttyālokanyāsāt sūkṣmavyavahitaviprakṛṣṭajñānam ',
 'bhuvanajñānaṃ sūrye saṃyamāt ',
 'candre tārāvyūhajñānam ',
 'dhruve tadgatijñānam ',
 'nābhicakre kāyavyūhajñānam ',
 'kaṇṭhakūpe kṣutpipāsānivṛttiḥ ',
 'kūrmanāḍyāṃ sthairyam ',
 'mūrdhajyotiṣi siddhadarśanam ',
 'prātibhād vā sarvam ',
 'hṛdaye cittasaṃvit ',
 'sattvapuruṣayor atyantāsaṃkīrṇayoḥ pratyayāviśeṣo bhogaḥ parārthāt svārthasaṃyamāt puruṣajñānam ',
 'tataḥ prātibhaśrāvaṇavedanādarśāsvādavārtā jāyante ',
 'te samādhāv upasargā vyutthāne siddhayaḥ ',
 'bandhakāraṇaśaithilyāt pracārasaṃvedanāc ca cittasya paraśarīrāveśaḥ ',
 'udānajayāj jalapaṅkakaṇṭakādiṣv asaṅga utkrāntiś ca ',
 'samānajayāj jvalanam ',
 'śrotrākāśayoḥ saṃbandhasaṃyamād divyaṃ śrotram ',
 'kāyākāśayoḥ saṃbandhasaṃyamāl laghutūlasamāpatteś cākāśagamanam ',
 'bahir akalpitā vṛttir mahāvidehā tataḥ prakāśāvaraṇakṣayaḥ ',
 'sthūlasvarūpasūkṣmānvayārthavattvasaṃyamād bhūtajayaḥ ',
 "tato 'ṇimādiprādurbhāvaḥ kāyasaṃpat taddharmānabhighātaś ca ",
 'rūpalāvaṇyabalavajrasaṃhananatvāni kāyasaṃpat ',
 'grahaṇasvarūpāsmitānvayārthavattvasaṃyamād indriyajayaḥ ',
 'tato manojavitvaṃ vikaraṇabhāvaḥ pradhānajayaś ca ',
 'sattvapuruṣānyatākhyātimātrasya sarvabhāvādhiṣṭhātṛtvaṃ sarvajñātṛtvaṃ ca ',
 'tadvairāgyād api doṣabījakṣaye kaivalyam ',
 'sthānyupanimantraṇe saṅgasmayākaraṇaṃ punar aniṣṭaprasaṅgāt ',
 'kṣaṇatatkramayoḥ saṃyamād vivekajaṃ jñānam ',
 'jātilakṣaṇadeśair anyatānavacchedāt tulyayos tataḥ pratipattiḥ ',
 'tārakaṃ sarvaviṣayaṃ sarvathāviṣayam akramaṃ ceti vivekajaṃ jñānam ',
 'sattvapuruṣayoḥ śuddhisāmye kaivalyam iti ',
 'janmauṣadhimantratapaḥsamādhijāḥ siddhayaḥ ',
 'jātyantarapariṇāmaḥ prakṛtyāpūrāt ',
 'nimittam aprayojakaṃ prakṛtīnāṃ varaṇabhedas tu tataḥ kṣetrikavat ',
 'nirmāṇacittāny asmitāmātrāt ',
 'pravṛttibhede prayojakaṃ cittam ekam anekeṣām ',
 'tatra dhyānajam anāśayam ',
 'karmāśuklākṛṣṇaṃ yoginas trividham itareṣām ',
 'tatas tadvipākānuguṇānām evābhivyaktir vāsanānām ',
 'jātideśakālavyavahitānām apy ānantaryaṃ smṛtisaṃskārayor ekarūpatvāt ',
 'tāsām anāditvaṃ cāśiṣo nityatvāt ',
 'hetuphalāśrayālambanaiḥ saṃgṛhītatvād eṣām abhāve tadabhāvaḥ ',
 "atītānāgataṃ svarūpato 'sty adhvabhedād dharmāṇām ",
 'te vyaktasūkṣmā guṇātmānaḥ ',
 'pariṇāmaikatvād vastutattvam ',
 'vastusāmye cittabhedāt tayor vibhaktaḥ panthāḥ ',
 'na caikacittatantraṃ vastu tadapramāṇakaṃ tadā kiṃ syāt ',
 'taduparāgāpekṣitvāc cittasya vastu jñātājñātam ',
 'sadā jñātāś cittavṛttayas tatprabhoḥ puruṣasyāpariṇāmitvāt ',
 'na tat svābhāsaṃ dṛśyatvāt ',
 'ekasamaye cobhayānavadhāraṇam ',
 'cittāntaradṛśye buddhibuddher atiprasaṅgaḥ smṛtisaṃkaraś ca ',
 'citer apratisaṃkramāyās tadākārāpattau svabuddhisaṃvedanam ',
 'draṣṭṛdṛśyoparaktaṃ cittaṃ sarvārtham ',
 'tad asaṃkhyeyavāsanābhiś citram api parārthaṃ saṃhatyakāritvāt ',
 'viśeṣadarśina ātmabhāvabhāvanānivṛttiḥ ',
 'tadā vivekanimnaṃ kaivalyaprāgbhāraṃ cittam ',
 'tacchidreṣu pratyayāntarāṇi saṃskārebhyaḥ ',
 'hānam eṣāṃ kleśavad uktam ',
 "prasaṃkhyāne 'py akusīdasya sarvathā vivekakhyāter dharmameghaḥ samādhiḥ ",
 'tataḥ kleśakarmanivṛttiḥ ',
 'tadā sarvāvaraṇamalāpetasya jñānasyānantyāj jñeyam alpam ',
 'tataḥ kṛtārthānāṃ pariṇāmakramasamāptir guṇānām ',
 'kṣaṇapratiyogī pariṇāmāparāntanirgrāhyaḥ kramaḥ ',
 'puruṣārthaśūnyānāṃ guṇānāṃ pratiprasavaḥ kaivalyaṃ svarūpapratiṣṭhā vā citiśaktir iti ']